"""
Export a trained TributeValueNetwork checkpoint to ONNX, verifying the graph
against PyTorch before writing it.
"""
import hashlib
import os
import torch
from ValueNetwork import TributeValueNetwork
from StateParser import NODE_DIM, GLOBAL_DIM

# Minimal stand-in for a PyG Batch, used only to run base_model during
# verification. The exported graph takes no batch index.
class MockBatch:
    def __init__(self, x, batch, u):
        self.x = x
        self.batch = batch
        self.u = u

# Exports a plain per-node mean instead of global_mean_pool. At opset 14
# global_mean_pool lowers to a ScatterElements with no reduction attribute,
# which replaces rather than averages. The agent always evaluates a single
# graph, so the plain mean is equivalent.
class ONNXWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.node_encoder = base_model.node_encoder
        self.global_encoder = base_model.global_encoder
        self.evaluator = base_model.evaluator

    def forward(self, x, u):
        h = self.node_encoder(x)
        pooled = h.mean(dim=0, keepdim=True)  # -> ReduceMean, dynamic-safe
        g = self.global_encoder(u)
        return self.evaluator(torch.cat([pooled, g], dim=1))


# Agreement tolerance |torch - onnx| <= VERIFY_ATOL + VERIFY_RTOL * |torch|, as
# in numpy.allclose. The output is a raw logit that can reach the tens, where a
# single float32 ULP exceeds 1e-5, so a fixed absolute bound rejects correct
# exports. rtol=1e-6 is about 10x float32 epsilon; a real graph error, such as
# the wrong pooling, is of order 1.
VERIFY_ATOL = 1e-5
VERIFY_RTOL = 1e-6

# Fixed seed for the verification inputs, so a pass or failure is reproducible.
VERIFY_SEED = 12345

# Weights with |w| < FLUSH_THRESHOLD are zeroed in the exported copy. The
# threshold sits well above float32's subnormal boundary (~1.18e-38) because
# near-subnormal weights still produce subnormal intermediates, which take
# onnxruntime off its fast path on x86. On the cluster, zeroing only true
# subnormals left seed_00 at 130 us per inference against 52.4 us at this
# threshold; 1e-20 and 1e-12 also changed no output. Apple Silicon flushes
# denormals in hardware, so the slowdown does not appear there.
FLUSH_THRESHOLD = 1e-30

# float32 subnormal boundary, for the diagnostic breakdown only.
FLOAT32_TINY = 1.1754943508222875e-38


def flush_denormals(state_dict, threshold=FLUSH_THRESHOLD):
    """Return a copy of state_dict with every float entry |w| < threshold set to
    zero, plus per-tensor statistics. The checkpoint itself is not modified.

    Returns (flushed_state_dict, rows, totals) where rows is a list of
    (name, zeroed, subnormal, numel) and totals is (zeroed, subnormal, numel).
    """
    flushed = {}
    rows = []
    tot_zeroed = tot_subnormal = tot_numel = 0

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            flushed[name] = tensor
            continue
        t = tensor.detach().clone()
        magnitude = t.abs()
        nonzero = t != 0
        # Count only entries the flush changes; exact zeros are left out.
        below = (magnitude < threshold) & nonzero
        subnormal = (magnitude < FLOAT32_TINY) & nonzero

        n_zeroed = int(below.sum().item())
        n_subnormal = int(subnormal.sum().item())
        t[below] = 0.0

        flushed[name] = t
        rows.append((name, n_zeroed, n_subnormal, t.numel()))
        tot_zeroed += n_zeroed
        tot_subnormal += n_subnormal
        tot_numel += t.numel()

    return flushed, rows, (tot_zeroed, tot_subnormal, tot_numel)


def print_flush_report(rows, totals, threshold):
    tot_zeroed, tot_subnormal, tot_numel = totals
    print(f"Flushing weights with |w| < {threshold:g} to zero "
          f"(subnormal boundary is {FLOAT32_TINY:.3e}):")
    print(f"  {'tensor':<28}{'params':>10}{'zeroed':>10}{'of which subnormal':>21}")
    for name, n_zeroed, n_subnormal, numel in rows:
        print(f"  {name:<28}{numel:>10,}{n_zeroed:>10,}{n_subnormal:>21,}")
    pct = (100.0 * tot_zeroed / tot_numel) if tot_numel else 0.0
    print(f"  {'TOTAL':<28}{tot_numel:>10,}{tot_zeroed:>10,}{tot_subnormal:>21,}"
          f"   ({pct:.3f}% zeroed)")
    if tot_zeroed == 0:
        print("  Nothing to flush -- this checkpoint has no weights below the threshold.")
        print("  Expected for anything trained on hardware that flushes denormals itself.")


def verify_export(base_model, onnx_filename):
    import onnxruntime as ort
    base_model.eval()

    # A local generator, so importing this module does not reseed the global RNG.
    gen = torch.Generator()
    gen.manual_seed(VERIFY_SEED)

    # Single-threaded, as in DeepSetsCore.cs's evaluator. Left to size itself,
    # onnxruntime starts a thread per machine core and hits affinity errors in a
    # smaller SLURM allocation.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_filename, opts, providers=["CPUExecutionProvider"])

    worst_abs = 0.0
    worst_rel = 0.0
    failures = []
    for n in [1, 2, 3, 5, 15, 16, 25, 40, 60]:
        x = torch.randn(n, NODE_DIM, dtype=torch.float32, generator=gen)
        u = torch.randn(1, GLOBAL_DIM, dtype=torch.float32, generator=gen)
        with torch.no_grad():
            ref = base_model(
                MockBatch(x, torch.zeros(n, dtype=torch.int64), u)
            ).item()
        got = float(sess.run(None, {
            "node_features": x.numpy(),
            "global_features": u.numpy(),
        })[0].reshape(-1)[0])

        abs_d = abs(ref - got)
        # Relative error is undefined at 0; the tolerance reduces to atol there.
        rel_d = abs_d / abs(ref) if ref != 0.0 else float("nan")
        tol = VERIFY_ATOL + VERIFY_RTOL * abs(ref)
        ok = abs_d <= tol

        worst_abs = max(worst_abs, abs_d)
        if rel_d == rel_d:  # not NaN
            worst_rel = max(worst_rel, rel_d)

        rel_txt = f"{rel_d:.3e}" if rel_d == rel_d else "n/a"
        print(f"  n={n:>3}  torch={ref:+.6f}  onnx={got:+.6f}  "
              f"abs={abs_d:.3e}  rel={rel_txt}  tol={tol:.3e}{'' if ok else '   <-- FAIL'}")
        if not ok:
            failures.append((n, ref, got, abs_d, rel_d, tol))

    if failures:
        detail = "\n".join(
            f"    n={n}: torch={ref:+.6f} onnx={got:+.6f} abs={abs_d:.3e} "
            f"rel={rel_d:.3e} > tol={tol:.3e}"
            for n, ref, got, abs_d, rel_d, tol in failures)
        raise RuntimeError(
            f"EXPORT VERIFICATION FAILED for {onnx_filename}: "
            f"{len(failures)} of 9 node counts outside tolerance "
            f"(atol={VERIFY_ATOL:.1e}, rtol={VERIFY_RTOL:.1e}).\n{detail}\n"
            f"  A relative error near 1e-7 is float32 rounding and means the "
            f"TOLERANCE is wrong, not the export.\n"
            f"  A relative error orders of magnitude above that means the graph "
            f"is wrong -- check the pooling op first (see ONNXWrapper above).")

    print(f"Export verified: max abs diff {worst_abs:.3e}, max rel diff {worst_rel:.3e} "
          f"(atol={VERIFY_ATOL:.1e}, rtol={VERIFY_RTOL:.1e})")


def export_model(checkpoint_path="best_model.pth", onnx_filename="DeepSetsValueNetwork.onnx",
                 flush=True):
    print("=== EXPORTING SAKKIRINA TO ONNX ===")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # base_model stays unflushed, so verification also confirms that the flush
    # changed no output.
    base_model = TributeValueNetwork(node_in_dim=NODE_DIM, global_in_dim=GLOBAL_DIM)
    base_model.load_state_dict(checkpoint)
    base_model.eval()  # disable dropout and batchnorm training behaviour

    # The flushed weights go in a separate instance: ONNXWrapper keeps references
    # to the submodules it is given, so flushing through it would flush
    # base_model too.
    if flush:
        print()
        flushed_state, rows, totals = flush_denormals(checkpoint, FLUSH_THRESHOLD)
        print_flush_report(rows, totals, FLUSH_THRESHOLD)
        print()
        export_source = TributeValueNetwork(node_in_dim=NODE_DIM, global_in_dim=GLOBAL_DIM)
        export_source.load_state_dict(flushed_state)
        export_source.eval()
    else:
        print()
        print("--no-flush: exporting the checkpoint's weights untouched.")
        print("  On x86-trained checkpoints this produces a model that is correct but "
              "far slower")
        print("  at inference -- roughly 17x was measured for seed_00 ON x86; Apple Silicon")
        print("  flushes denormals in hardware and shows no gap. Correct choice when "
              "reproducing")
        print("  an export made before flushing existed, such as the shipped model.")
        print()
        export_source = base_model

    wrapped_model = ONNXWrapper(export_source)

    # Example inputs for tracing: 15 nodes, and one global vector.
    num_nodes = 15

    dummy_x = torch.randn(num_nodes, NODE_DIM, dtype=torch.float32)

    dummy_u = torch.randn(1, GLOBAL_DIM, dtype=torch.float32)

    # Export to a temporary file beside the target and rename it into place only
    # once verification passes, so a failed export leaves no .onnx behind. Same
    # directory so os.replace is atomic; pid-tagged so concurrent exports cannot
    # collide.
    tmp_filename = f"{onnx_filename}.tmp{os.getpid()}"

    torch.onnx.export(
        wrapped_model,
        (dummy_x, dummy_u),  # raw tensor inputs, no batch index
        tmp_filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['node_features', 'global_features'],
        output_names=['win_probability'],
        # The node count varies from state to state.
        dynamic_axes={
            'node_features': {0: 'num_nodes'},
            'win_probability': {0: 'batch_size'}
        }
    )

    print(f"Exported to a temporary file, verifying before writing {onnx_filename} ...")
    try:
        verify_export(base_model, tmp_filename)
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

    print(f"✅ Successfully exported to {onnx_filename}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export a trained TributeValueNetwork checkpoint to ONNX.")
    parser.add_argument("--checkpoint", default="best_model.pth", help="Path to the .pth checkpoint to export")
    parser.add_argument("--out", default="DeepSetsValueNetwork.onnx", help="Output .onnx path")
    parser.add_argument("--no-flush", action="store_true",
                        help=f"Do NOT zero weights with |w| < {FLUSH_THRESHOLD:g} before export. "
                             f"Flushing is on by default because leaving x86-trained denormals "
                             f"in place costs roughly 17x at inference on x86. Pass this when "
                             f"reproducing an export made before flushing existed -- the shipped "
                             f"model, for one. Flushed and unflushed exports of the same "
                             f"checkpoint give bit-identical outputs on real states; only the "
                             f"bytes differ.")
    args = parser.parse_args()
    export_model(checkpoint_path=args.checkpoint, onnx_filename=args.out,
                 flush=not args.no_flush)
