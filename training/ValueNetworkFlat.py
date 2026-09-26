"""
Plain MLPs over the padded flat vector -- the architecture half of the DeepSets
ablation.

TWO CONFIGURATIONS, ANSWERING TWO DIFFERENT OBJECTIONS.

  matched   12,691 -> 5 -> 128 -> 64 -> 1        72,549 parameters
            Equal capacity (99.3% of the DeepSets model's 73,089). Answers:
            does the set structure help when the two models are allowed the
            same number of weights?

  wide      12,691 -> 128 -> 128 -> 64 -> 1   1,649,409 parameters
            22.6x the DeepSets model. Answers: does the set structure help even
            when the flat model is given far more capacity than the model it is
            being compared against? If the flat model still loses here, the
            result is much stronger, and it forecloses the obvious objection
            that the matched configuration was starved into losing.

WHY "matched" IS A 5-UNIT FIRST LAYER, AND WHY THAT IS NOT A BUG. At a
12,691-dim input the first layer alone costs 12,691 weights per unit, so the
parameter budget caps it at 73,089 / 12,692 = 5.75 units even if the rest of the
network were free. Five is what fits. This is a real and severe bottleneck and
it is stated rather than engineered around, because engineering around it means
either breaking the parameter match or truncating the input -- and truncation
would remove enemy hand-and-draw cards specifically, turning an architecture
ablation into an architecture-plus-information one (see StateParserFlat).

It is also worth being precise about what the bottleneck is NOT evidence for.
Choosing a smaller MAX_NODES buys almost nothing: the ceiling is 7 units at 96
nodes, 11 at 60, 12 at 48. There is no cap at which a parameter-matched flat
model over this input is not a single-digit-to-low-double-digit bottleneck. The
narrowness is a consequence of flattening 128x99 inputs on a 73k budget, which
is precisely the cost of not having a shared per-card encoder. The "wide"
configuration exists so that this cost can be paid off and the structural
question asked separately.

The head (-> 128 -> 64 -> 1) is deliberately identical to the DeepSets model's
evaluator tail, so the two networks differ in how they get to a 128-dim
representation and in nothing after it.

THROUGHPUT: THE ARITHMETIC WAS A HYPOTHESIS, AND IT OVER-PREDICTED BY 10x.
Counting multiply-accumulates, the DeepSets model runs its node encoder once per
card: 29,056 MACs per node plus 43,456 fixed, so about 1,002,000 MACs at the
median 33-node state and 3,762,624 at a 128-node one. Either flat model's cost
is constant in the node count, and "matched" comes to about 72,000 MACs --
roughly 14x fewer at the median. That was a reason to EXPECT the flat model to
be faster. It was not a measurement, and it was wrong about the size of the
effect.

Measured (tools/compare_onnx_models.py, 5,000 real states, single-threaded):

    DeepSets         74.9 us mean,  70.7 us median
    flat-matched     51.9 us mean,  49.7 us median     1.45x FASTER
    flat-wide       197.0 us mean, 195.6 us median     2.47x SLOWER

So a 14x MAC advantage becomes 1.45x wall clock. The gap is the point: a
12,691->5 matvec streams 63,455 weights to produce five numbers and is entirely
memory-bound, so its MACs are nearly free but its loads are not, while 33
batched 99->128 rows is a shape onnxruntime is good at. Arithmetic intensity,
not arithmetic, is what decides this. "wide" loses outright because 1,649,409
weights is 6.6 MB and past any useful cache residency -- more capacity is a
throughput cost here even before it is an accuracy question.

Those are Apple Silicon numbers (M1, x86_64 Python under Rosetta). The denormal
warning in export_to_onnx.py does not apply -- none of these models carry
subnormal weights -- but GEMM-shape efficiency is still host-specific, so
re-measure on the cluster before quoting a ratio in the paper.
"""

import torch
import torch.nn as nn

from StateParserFlat import FLAT_DIM, batch_pad_and_flatten

# name -> (h1, h2, h3). See the module docstring for what each one answers.
FLAT_CONFIGS = {
    "matched": (5, 128, 64),
    "wide": (128, 128, 64),
}


class TributeValueNetworkFlat(nn.Module):
    """A plain MLP over the [FLAT_DIM] vector. Emits a raw logit, like
    TributeValueNetwork -- the sigmoid lives in the caller."""

    def __init__(self, in_dim=FLAT_DIM, h1=5, h2=128, h3=64):
        super().__init__()
        self.in_dim = in_dim
        self.widths = (h1, h2, h3)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, z):
        """z: [B, FLAT_DIM] -> [B, 1]."""
        return self.mlp(z)


class FlatGraphAdapter(nn.Module):
    """Training-only shim: PyG Batch -> flat matrix -> MLP.

    train_local.py's loop calls model(batch) with a PyG Batch and reads
    batch.u[:, PRESTIGE_CLOCK_GLOBAL_INDEX] for its bucketed metrics. This lets
    the flat models run through that loop untouched, so both arms of the
    ablation share one training implementation rather than a copy of it.

    Not part of the exported artefact. export_flat_to_onnx.py traces the inner
    TributeValueNetworkFlat with padding expressed in ONNX ops, so nothing in
    this class reaches the .onnx file.
    """

    def __init__(self, flat_model):
        super().__init__()
        self.flat = flat_model

    def forward(self, data):
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        num_graphs = data.u.shape[0]
        z = batch_pad_and_flatten(data.x, batch, data.u, num_graphs=num_graphs)
        return self.flat(z)


def build_flat_model(arch="matched"):
    """arch -> (FlatGraphAdapter for training, inner TributeValueNetworkFlat)."""
    if arch not in FLAT_CONFIGS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(FLAT_CONFIGS)}")
    h1, h2, h3 = FLAT_CONFIGS[arch]
    inner = TributeValueNetworkFlat(in_dim=FLAT_DIM, h1=h1, h2=h2, h3=h3)
    return FlatGraphAdapter(inner), inner


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from ValueNetwork import TributeValueNetwork

    deepsets = count_parameters(TributeValueNetwork())
    print(f"{'config':<10}{'input':>9}  {'shape':<32}{'parameters':>12}{'vs DeepSets':>13}")
    print(f"{'deepsets':<10}{'set':>9}  {'99->128->128 pooled, 256->128->64->1':<32}"
          f"{deepsets:>12,}{'1.00x':>13}")
    for name in FLAT_CONFIGS:
        _, inner = build_flat_model(name)
        n = count_parameters(inner)
        shape = f"{FLAT_DIM}->" + "->".join(str(w) for w in inner.widths) + "->1"
        print(f"{name:<10}{FLAT_DIM:>9,}  {shape:<32}{n:>12,}{n / deepsets:>12.2f}x")
