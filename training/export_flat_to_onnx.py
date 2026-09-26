"""
Export a trained flat-MLP checkpoint to ONNX.

Everything safety-related is imported from export_to_onnx rather than restated:
the combined-tolerance verification constants, the seeded verification inputs,
the denormal flush and its report. Only the graph differs. A second copy of any
of that would be a second thing to keep correct, and the ONE thing this script
must not do is put an unverified model on disk -- which is how seeds 3 and 4
ended up as plausible-looking .onnx files nothing had ever checked.

THE EXPORTED GRAPH TAKES THE SAME INPUTS AS THE DEEPSETS MODEL.
Inputs are (node_features [n, 99], global_features [1, 19]) with n dynamic, and
the output is win_probability [1, 1] -- byte-for-byte the same signature
DeepSetsValueNetwork.onnx presents. The padding, flattening and concatenation
happen INSIDE the graph.

Two things follow, and both are deliberate:

  1. tools/compare_onnx_models.py works on these files with no changes, so the
     throughput half of the ablation is measured by the same tool, on the same
     real states, as everything else.

  2. If either flat model ever earns a bot, it may not need a C# encoder at all:
     an agent can feed it through the unchanged FeatureExtractor in
     DeepSetsCore.cs. StateParserFlat exists as the reference definition of the
     layout and as the training-time path, not necessarily as something that has
     to be ported.

It also means the padding cost is inside the measurement, which is correct. A
real agent would pay it -- zeroing 12,672 floats on every evaluation is work the
set encoder never does -- and excluding it would flatter the flat model against
the one architecture it is being compared to.

WHY VERIFICATION USES ITS OWN LOOP. export_to_onnx.verify_export drives a
MockBatch through TributeValueNetwork's forward, which this model does not have.
The tolerance, the seeding discipline and the failure message are the same; only
the two lines that produce the reference value differ.
"""

import argparse
import hashlib
import os

import torch

from StateParser import NODE_DIM, GLOBAL_DIM
from StateParserFlat import FLAT_DIM, FLAT_NODE_DIM, MAX_NODES, pad_and_flatten
from ValueNetworkFlat import FLAT_CONFIGS, TributeValueNetworkFlat, count_parameters
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

    Slice first, then pad. F.pad with a negative amount is not a crop in ONNX,
    so a state with more than MAX_NODES nodes has to be cut before the pad
    amount is computed rather than relying on the pad to do it. No state in the
    corpus reaches that branch -- 128 IS the measured maximum -- but a graph
    that silently produced a wrong-shaped tensor on one would be worse than one
    that truncates.

    CONCAT WITH ZEROS RATHER THAN F.pad, because the exported graph is a
    measurement instrument here. torch.nn.functional.pad on a 2-D tensor lowers
    to Transpose -> Pad -> Transpose plus the index arithmetic to build the pads
    vector: 29 nodes in the graph, against 17 for the concat, and measurably
    slower -- 45.9 us against 43.3 us median at a 33-node state on this host.
    Both produce identical output. Charging the flat model 6% for an avoidable
    Transpose and then reporting the total as an architectural property would be
    wrong, so the padding is written the efficient way and the remaining cost is
    real.
    """

    def __init__(self, flat_model):
        super().__init__()
        self.mlp = flat_model.mlp

    def forward(self, x, u):
        x = x[:MAX_NODES]
        pad_rows = MAX_NODES - x.shape[0]
        x = torch.cat([x, x.new_zeros(pad_rows, NODE_DIM)], dim=0)
        flat = x.reshape(1, FLAT_NODE_DIM)
        return self.mlp(torch.cat([flat, u], dim=1))


def verify_flat_export(base_model, onnx_filename):
    """Same contract as export_to_onnx.verify_export: combined tolerance
    atol + rtol*|torch|, fixed seed, single-threaded session, nothing written
    unless every node count passes.

    The node counts swept include MAX_NODES and MAX_NODES+1. The last one is the
    truncation branch, and it is checked against pad_and_flatten -- which
    truncates the same way -- so the graph and the Python encoder agree even
    where the corpus never goes.

    EXPECT A LARGER RELATIVE ERROR HERE THAN export_to_onnx.py SEES, and do not
    read it as a bug. That file's guidance is that a relative error near 1e-7 is
    float32 rounding; it says so because its longest dot products are 256 terms.
    This model's first layer sums 12,691 terms, and accumulated rounding grows
    roughly as sqrt(n)*eps -- about sqrt(12691)*1.2e-7 = 1.3e-5 -- so a relative
    error of ~1e-5 is the FLOOR here, not a warning sign. Measured on the
    trained matched checkpoint: 1.105e-05 relative at n=128, which is ~90x
    float32 epsilon and entirely expected.

    The combined tolerance absorbs this because atol dominates wherever the
    output is small, and the margins stay comfortable in practice -- the worst
    observed was 8.583e-06 against a 2.586e-05 tolerance, 33% of budget. A real
    graph bug (wrong pad, wrong flatten order, transposed weights) produces
    errors of order 1, nowhere near either number.
    """
    import onnxruntime as ort

    base_model.eval()

    # A LOCAL generator, not torch.manual_seed: this module gets imported, and
    # a verification helper that quietly reseeds the global RNG would perturb
    # whatever else in the process is drawing from it.
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
            ref = base_model(pad_and_flatten(x, u).unsqueeze(0)).item()
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


def export_flat_model(checkpoint_path, onnx_filename, arch="matched", flush=True):
    print(f"=== EXPORTING FLAT-MLP ({arch}) TO ONNX ===")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # train_flat.py trains a FlatGraphAdapter wrapping the MLP, so its
    # state_dict keys are prefixed "flat.". Strip that: the adapter is training
    # plumbing and nothing in the exported graph corresponds to it.
    if any(k.startswith("flat.") for k in checkpoint):
        checkpoint = {k[len("flat."):]: v for k, v in checkpoint.items()
                      if k.startswith("flat.")}

    h1, h2, h3 = FLAT_CONFIGS[arch]

    # base_model is the checkpoint EXACTLY as trained, and stays that way -- it
    # is what verify_flat_export measures the exported graph against, so a
    # flushed ONNX that still matches it to within tolerance is the statement we
    # want out of the check.
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

    wrapped_model = FlatONNXWrapper(export_source)

    # 15 nodes, matching export_to_onnx.py. The trace is over a dynamic axis, so
    # the particular value only has to be a count the graph handles normally --
    # not MAX_NODES, which would trace the pad at zero rows and could constant-
    # fold the Pad away.
    dummy_x = torch.randn(15, NODE_DIM, dtype=torch.float32)
    dummy_u = torch.randn(1, GLOBAL_DIM, dtype=torch.float32)

    # VERIFY BEFORE PUBLISHING: temp file in the same directory (so os.replace
    # is atomic), pid-tagged (so concurrent exports cannot collide), renamed
    # into place only once verification passes.
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
        verify_flat_export(base_model, tmp_filename)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or a SLURM timeout
        # mid-verification must not leave the temp file behind either.
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
    parser.add_argument("--arch", default="matched", choices=sorted(FLAT_CONFIGS),
                        help="Must match what the checkpoint was trained with; the "
                             "load_state_dict below fails loudly if it does not.")
    parser.add_argument("--no-flush", action="store_true",
                        help=f"Do NOT zero weights with |w| < {FLUSH_THRESHOLD:g} before "
                             f"export. Flushing is on by default -- see export_to_onnx.py.")
    args = parser.parse_args()

    export_flat_model(checkpoint_path=args.checkpoint, onnx_filename=args.out,
                      arch=args.arch, flush=not args.no_flush)
