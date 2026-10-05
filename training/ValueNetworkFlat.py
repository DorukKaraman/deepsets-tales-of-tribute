"""
Plain MLPs over the padded flat vector, for the DeepSets ablation.

  matched         12,691 -> 5 -> 128 -> 64 -> 1        72,549 parameters
  wide            12,691 -> 128 -> 128 -> 64 -> 1   1,649,409 parameters
  matched_sorted  as matched, on canonically sorted node rows

matched is held to roughly the DeepSets model's 73,089 parameters, which at a
12,691-dim input leaves room for only five first-layer units. wide removes that
limit; matched_sorted makes the input permutation-invariant. The head
(-> 128 -> 64 -> 1) is the same as the DeepSets evaluator's. Results are in
REPRODUCE.md section 8.
"""

import torch
import torch.nn as nn

from StateParserFlat import FLAT_DIM, batch_pad_and_flatten

# name -> (h1, h2, h3) hidden-layer widths.
FLAT_CONFIGS = {
    "matched": (5, 128, 64),
    "wide": (128, 128, 64),
    "matched_sorted": (5, 128, 64),
}

# Archs whose node rows are sorted into canonical order before flattening.
# matched_sorted has matched's widths, so the two differ only in the sort.
SORTED_ARCHS = {"matched_sorted"}


class TributeValueNetworkFlat(nn.Module):
    """A plain MLP over the [FLAT_DIM] vector, returning a raw logit; the
    caller applies the sigmoid."""

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
    """Training-only wrapper: PyG Batch -> flat matrix -> MLP.

    Lets train_local.py's loop, which passes a PyG Batch, drive the flat
    models. It is not exported: export_flat_to_onnx.py traces the inner MLP.
    """

    def __init__(self, flat_model, sort=False):
        super().__init__()
        self.flat = flat_model
        self.sort = sort

    def forward(self, data):
        batch = data.batch if getattr(data, "batch", None) is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        num_graphs = data.u.shape[0]
        z = batch_pad_and_flatten(data.x, batch, data.u, num_graphs=num_graphs,
                                  sort=self.sort)
        return self.flat(z)


def build_flat_model(arch="matched"):
    """arch -> (FlatGraphAdapter for training, inner TributeValueNetworkFlat)."""
    if arch not in FLAT_CONFIGS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(FLAT_CONFIGS)}")
    h1, h2, h3 = FLAT_CONFIGS[arch]
    inner = TributeValueNetworkFlat(in_dim=FLAT_DIM, h1=h1, h2=h2, h3=h3)
    return FlatGraphAdapter(inner, sort=arch in SORTED_ARCHS), inner


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from ValueNetwork import TributeValueNetwork

    deepsets = count_parameters(TributeValueNetwork())
    print(f"{'config':<16}{'input':>9}  {'shape':<38}{'rows':<11}"
          f"{'parameters':>12}{'vs DeepSets':>13}")
    print(f"{'deepsets':<16}{'set':>9}  {'99->128->128 pooled, 256->128->64->1':<38}"
          f"{'n/a (set)':<11}{deepsets:>12,}{'1.00x':>13}")
    for name in FLAT_CONFIGS:
        _, inner = build_flat_model(name)
        n = count_parameters(inner)
        shape = f"{FLAT_DIM}->" + "->".join(str(w) for w in inner.widths) + "->1"
        rows = "canonical" if name in SORTED_ARCHS else "as emitted"
        print(f"{name:<16}{FLAT_DIM:>9,}  {shape:<38}{rows:<11}"
              f"{n:>12,}{n / deepsets:>12.2f}x")
