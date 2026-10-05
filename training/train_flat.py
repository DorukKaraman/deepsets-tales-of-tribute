"""
Train a flat-MLP baseline for the DeepSets ablation.

The loop, optimizer, seeding, metrics and checkpointing are those of
train_local.train_model; this only chooses the model, so a given --seed gives
every arm the same batches in the same order. Arguments and outputs match
train_local.py, and run_config.json also records the arch and layer widths.

    python training/train_flat.py --arch matched \\
        --train-dir $SPLIT/train --val-dir $SPLIT/val \\
        --epochs 3 --batch-size 256 --lr 5e-4 --seed 0 --out-dir $OUT

  --arch matched         12,691 -> 5 -> 128 -> 64 -> 1        72,549 params
  --arch wide            12,691 -> 128 -> 128 -> 64 -> 1   1,649,409 params
  --arch matched_sorted  as matched, with the node rows in canonical order

See ValueNetworkFlat for what each arm is meant to test.
"""

import argparse

from StateParserFlat import FLAT_DIM, MAX_NODES
from ValueNetworkFlat import (FLAT_CONFIGS, SORTED_ARCHS, build_flat_model,
                              count_parameters)
from train_local import train_model


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", required=True,
                        help="Directory of train shards (tools/split_dataset.py's train/)")
    parser.add_argument("--val-dir", required=True,
                        help="Directory of val shards (tools/split_dataset.py's val/)")
    parser.add_argument("--arch", default="matched", choices=sorted(FLAT_CONFIGS),
                        help="matched = parameter-matched to DeepSets (72,549); "
                             "wide = 22.57x capacity (1,649,409); matched_sorted = "
                             "matched with canonically ordered rows. Run all three.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default=".",
                        help="Where to write checkpoints and training_metrics.json")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds torch, numpy, python random and the shuffle buffer. "
                             "The same seed gives the DeepSets run and both flat runs the "
                             "identical stream of batches.")
    args = parser.parse_args()

    h1, h2, h3 = FLAT_CONFIGS[args.arch]
    shape = f"{FLAT_DIM}->{h1}->{h2}->{h3}->1"

    # Built only to report the parameter count. train_model seeds before it
    # constructs the real model, so drawing from the RNG here is harmless.
    _, probe = build_flat_model(args.arch)
    n_params = count_parameters(probe)

    print(f"=== FLAT-MLP ABLATION ({args.arch}) ===")
    print(f"  input      : {FLAT_DIM:,} ({MAX_NODES} nodes x 99 + 19 global)")
    print(f"  shape      : {shape}")
    print(f"  row order  : {'canonical (sorted)' if args.arch in SORTED_ARCHS else 'as emitted'}")
    print(f"  parameters : {n_params:,}")
    print()

    train_model(
        args.train_dir, args.val_dir, args.epochs, args.batch_size, args.lr,
        args.num_workers, args.out_dir, args.seed,
        model_factory=lambda: build_flat_model(args.arch)[0],
        checkpoint_prefix=f"flat_{args.arch}_value_network",
        run_config_extra={
            "model": "flat_mlp",
            "arch": args.arch,
            "widths": [h1, h2, h3],
            "shape": shape,
            "max_nodes": MAX_NODES,
            "input_dim": FLAT_DIM,
            "canonical_row_order": args.arch in SORTED_ARCHS,
        },
    )


if __name__ == "__main__":
    main()
