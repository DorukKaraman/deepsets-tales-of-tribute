"""
Export a trained flat-MLP checkpoint to ONNX.

Uses export_to_onnx's tolerance, seeded inputs and denormal flush, and the same
write-to-temp, verify, rename sequence. The exported graph takes the same
inputs as the DeepSets model, (node_features [n, 99], global_features [1, 19]),
and pads, flattens and (for sorted archs) sorts internally, so
tools/compare_onnx_models.py and DeepSetsCore.cs's evaluator load it unchanged.

    python export_flat_to_onnx.py --checkpoint best_model.pth --out flat.onnx

--arch is read from the run_config.json beside the checkpoint if not given.
"""

import argparse
import hashlib
import os

import torch

from StateParser import NODE_DIM, GLOBAL_DIM
from StateParserFlat import (FLAT_DIM, FLAT_NODE_DIM, MAX_NODES, pad_and_flatten,
                             row_sort_key)
from ValueNetworkFlat import (FLAT_CONFIGS, SORTED_ARCHS, TributeValueNetworkFlat,
                              count_parameters)
from export_to_onnx import (
    FLUSH_THRESHOLD,
    VERIFY_ATOL,
    VERIFY_RTOL,
    VERIFY_SEED,
    flush_denormals,
    print_flush_report,
)


class FlatONNXWrapper(torch.nn.Module):
    """(node_features, global_features) -> pad -> flatten -> concat -> MLP.

    Rows beyond MAX_NODES are sliced off before padding, since a negative pad
    is not a crop in ONNX. Padding is a concat with zeros rather than F.pad,
    which on a 2-D tensor lowers to Transpose -> Pad -> Transpose and is slower.
    """

    def __init__(self, flat_model, sort=False):
        super().__init__()
        self.mlp = flat_model.mlp
        self.sort = sort

    def forward(self, x, u):
        x = x[:MAX_NODES]
        if self.sort:
            # Sort before padding, as in StateParserFlat.pad_and_flatten.
            # argsort lowers to TopK, which opset 14 supports.
            x = x[torch.argsort(row_sort_key(x), dim=0)]
        pad_rows = MAX_NODES - x.shape[0]
        x = torch.cat([x, x.new_zeros(pad_rows, NODE_DIM)], dim=0)
        flat = x.reshape(1, FLAT_NODE_DIM)
        return self.mlp(torch.cat([flat, u], dim=1))


def verify_flat_export(base_model, onnx_filename, sort=False):
    """Same checks as export_to_onnx.verify_export, against pad_and_flatten.

    The node counts include MAX_NODES + 1 to exercise truncation. Relative
    errors near 1e-5 are expected rather than suspicious: the first layer sums
    12,691 terms, so float32 rounding grows to about sqrt(12691) * 1.2e-7. A
    graph error is of order 1.
    """
    import onnxruntime as ort

    base_model.eval()

    # A local generator, so importing this module does not reseed the global RNG.
    gen = torch.Generator()
    gen.manual_seed(VERIFY_SEED)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_filename, opts, providers=["CPUExecutionProvider"])

    node_counts = [1, 2, 3, 5, 15, 25, 33, 60, 96, MAX_NODES, MAX_NODES + 1]
    worst_abs = 0.0
    worst_rel = 0.0
    failures = []

    for n in node_counts:
        x = torch.randn(n, NODE_DIM, dtype=torch.float32, generator=gen)
        u = torch.randn(1, GLOBAL_DIM, dtype=torch.float32, generator=gen)
        with torch.no_grad():
            ref = base_model(pad_and_flatten(x, u, sort=sort).unsqueeze(0)).item()
        got = float(sess.run(None, {
            "node_features": x.numpy(),
            "global_features": u.numpy(),
        })[0].reshape(-1)[0])

        abs_d = abs(ref - got)
        rel_d = abs_d / abs(ref) if ref != 0.0 else float("nan")
        tol = VERIFY_ATOL + VERIFY_RTOL * abs(ref)
        ok = abs_d <= tol

        worst_abs = max(worst_abs, abs_d)
        if rel_d == rel_d:  # not NaN
            worst_rel = max(worst_rel, rel_d)

        rel_txt = f"{rel_d:.3e}" if rel_d == rel_d else "n/a"
        note = "  (truncates)" if n > MAX_NODES else ""
        print(f"  n={n:>4}  torch={ref:+.6f}  onnx={got:+.6f}  "
              f"abs={abs_d:.3e}  rel={rel_txt}  tol={tol:.3e}"
              f"{'' if ok else '   <-- FAIL'}{note}")
        if not ok:
            failures.append((n, ref, got, abs_d, rel_d, tol))

    if failures:
        detail = "\n".join(
            f"    n={n}: torch={ref:+.6f} onnx={got:+.6f} abs={abs_d:.3e} "
            f"rel={rel_d:.3e} > tol={tol:.3e}"
            for n, ref, got, abs_d, rel_d, tol in failures)
        raise RuntimeError(
            f"EXPORT VERIFICATION FAILED for {onnx_filename}: "
            f"{len(failures)} of {len(node_counts)} node counts outside tolerance "
            f"(atol={VERIFY_ATOL:.1e}, rtol={VERIFY_RTOL:.1e}).\n{detail}\n"
            f"  A relative error around 1e-5 is float32 rounding over a "
            f"{FLAT_DIM}-term dot product and means the TOLERANCE is wrong, not "
            f"the export -- see this function's docstring for why the floor here "
            f"is ~100x what export_to_onnx.py sees.\n"
            f"  A relative error orders of magnitude above that means the graph "
            f"is wrong -- check the padding first: if only the large node counts "
            f"fail, the Pad or Slice is off by a row; if ALL of them fail, the "
            f"flatten order does not match StateParserFlat.pad_and_flatten.")

    print(f"Export verified: max abs diff {worst_abs:.3e}, max rel diff {worst_rel:.3e} "
          f"(atol={VERIFY_ATOL:.1e}, rtol={VERIFY_RTOL:.1e})")


def export_flat_model(checkpoint_path, onnx_filename, arch, flush=True):
    print(f"=== EXPORTING FLAT-MLP ({arch}) TO ONNX ===")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # train_flat.py saves a FlatGraphAdapter, whose keys carry a "flat." prefix.
    if any(k.startswith("flat.") for k in checkpoint):
        checkpoint = {k[len("flat."):]: v for k, v in checkpoint.items()
                      if k.startswith("flat.")}

    h1, h2, h3 = FLAT_CONFIGS[arch]

    # base_model stays unflushed, so verification also confirms that the flush
    # changed no output.
    base_model = TributeValueNetworkFlat(in_dim=FLAT_DIM, h1=h1, h2=h2, h3=h3)
    base_model.load_state_dict(checkpoint)
    base_model.eval()
    print(f"  {FLAT_DIM}->{h1}->{h2}->{h3}->1, {count_parameters(base_model):,} parameters")

    if flush:
        print()
        flushed_state, rows, totals = flush_denormals(checkpoint, FLUSH_THRESHOLD)
        print_flush_report(rows, totals, FLUSH_THRESHOLD)
        print()
        export_source = TributeValueNetworkFlat(in_dim=FLAT_DIM, h1=h1, h2=h2, h3=h3)
        export_source.load_state_dict(flushed_state)
        export_source.eval()
    else:
        print()
        print("--no-flush: exporting the checkpoint's weights untouched.")
        print("  On x86-trained checkpoints this produces a model that is correct but "
              "far slower")
        print("  at inference. Apple Silicon flushes denormals in hardware and shows no "
              "gap, so")
        print("  the difference is invisible on a Mac -- see export_to_onnx.py's "
              "FLUSH_THRESHOLD.")
        print()
        export_source = base_model

    wrapped_model = FlatONNXWrapper(export_source, sort=arch in SORTED_ARCHS)

    # 15 example nodes. Not MAX_NODES: tracing with no padding rows could fold
    # the padding step away.
    dummy_x = torch.randn(15, NODE_DIM, dtype=torch.float32)
    dummy_u = torch.randn(1, GLOBAL_DIM, dtype=torch.float32)

    # Export to a pid-tagged temp file beside the target (so os.replace is
    # atomic) and rename it into place only once verification passes.
    tmp_filename = f"{onnx_filename}.tmp{os.getpid()}"

    torch.onnx.export(
        wrapped_model,
        (dummy_x, dummy_u),
        tmp_filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['node_features', 'global_features'],
        output_names=['win_probability'],
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'win_probability': {0: 'batch_size'}
        }
    )

    print(f"Exported to a temporary file, verifying before writing {onnx_filename} ...")
    try:
        verify_flat_export(base_model, tmp_filename, sort=arch in SORTED_ARCHS)
    except BaseException:
        # BaseException, so an interrupt or SLURM timeout also removes the temp file.
        try:
            os.remove(tmp_filename)
            print(f"Verification failed -- removed {tmp_filename}, "
                  f"{onnx_filename} was not written.")
        except OSError:
            pass
        raise

    os.replace(tmp_filename, onnx_filename)

    file_size = os.path.getsize(onnx_filename)
    with open(onnx_filename, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"{onnx_filename}: size={file_size} bytes, sha256={sha256}")
    print(f"Successfully exported to {onnx_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a trained flat-MLP checkpoint to ONNX.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="best_model.pth",
                        help="Path to the .pth checkpoint to export")
    parser.add_argument("--out", default="FlatValueNetwork.onnx", help="Output .onnx path")
    parser.add_argument("--arch", default=None, choices=sorted(FLAT_CONFIGS),
                        help="Which flat arch this checkpoint was trained as. Read from "
                             "run_config.json beside the checkpoint if omitted; there is "
                             "deliberately NO default. 'matched' and 'matched_sorted' "
                             "have identical shapes, so load_state_dict cannot catch a "
                             "wrong value -- it would export a graph without the "
                             "canonical sort, and verification would pass, because the "
                             "reference it checks against would be wrong the same way.")
    parser.add_argument("--no-flush", action="store_true",
                        help=f"Do NOT zero weights with |w| < {FLUSH_THRESHOLD:g} before "
                             f"export. Flushing is on by default -- see export_to_onnx.py.")
    args = parser.parse_args()

    arch = args.arch
    if arch is None:
        cfg = os.path.join(os.path.dirname(os.path.realpath(args.checkpoint)),
                           "run_config.json")
        if os.path.isfile(cfg):
            import json
            with open(cfg) as f:
                arch = json.load(f).get("arch")
        if arch is None:
            raise SystemExit(
                f"ERROR: --arch not given and no usable 'arch' in {cfg}.\n"
                f"  The arch is not recoverable from the weights: 'matched' and "
                f"'matched_sorted' have identical widths and parameter counts, and "
                f"differ only in whether node rows are sorted before flattening.\n"
                f"  Guessing would export a graph missing the canonical sort, and the "
                f"export verification would still PASS, because it compares against a "
                f"PyTorch reference built from the same wrong guess.\n"
                f"  Pass --arch explicitly, or keep train_flat.py's run_config.json "
                f"beside the checkpoint.")
        print(f"--arch not given; using arch={arch!r} from {cfg}")
    if arch not in FLAT_CONFIGS:
        raise SystemExit(f"ERROR: unknown arch {arch!r}; choose from "
                         f"{', '.join(sorted(FLAT_CONFIGS))}")

    export_flat_model(checkpoint_path=args.checkpoint, onnx_filename=args.out,
                      arch=arch, flush=not args.no_flush)
