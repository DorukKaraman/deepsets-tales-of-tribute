"""
Train a flat-MLP baseline -- the ablation that tests whether the DeepSets
structure contributes, or whether the 99-dim card features alone carry the
result.

This is an ENTRY POINT, not a training implementation. The loop, optimizer,
scheduler, seeding, per-prestige-bucket metrics and checkpointing all come from
train_local.train_model; the only thing passed in is which model to build. That
matters for the ablation specifically: if the two arms had separate training
code, any difference in val loss could be a difference in how they were trained,
and the comparison would be worth nothing. They share one loop, one dataset
class, one shuffle buffer size and one seeding scheme, so a given --seed streams
the identical batches in the identical order to both.

    python training/train_flat.py --arch matched \\
        --train-dir $SPLIT/train --val-dir $SPLIT/val \\
        --epochs 3 --batch-size 256 --lr 5e-4 --seed 0 --out-dir $OUT

Arguments and artefacts match train_local.py exactly -- best_model.pth, a
per-epoch checkpoint, training_metrics.json, run_config.json -- so
scripts/slurm_train.sh's structure applies unchanged, with --arch added and the
export step pointed at export_flat_to_onnx.py. run_config.json additionally
records arch, widths, input dim and parameter count, so a directory of runs is
self-describing when it comes time to put the numbers in a table.

  --arch matched   12,691 -> 5 -> 128 -> 64 -> 1        72,549 params (0.99x DeepSets)
  --arch wide      12,691 -> 128 -> 128 -> 64 -> 1   1,649,409 params (22.57x)

Run BOTH. "matched" asks whether the set structure helps at equal capacity;
"wide" asks whether it helps even when the flat model has 22x the capacity, and
only "wide" answers the objection that the matched model was starved into
losing. See ValueNetworkFlat for why matched is a 5-unit first layer and why
that is a property of flattening a 12,691-dim input on a 73k budget rather than
a choice that could have been made differently.
"""

import argparse

from StateParserFlat import FLAT_DIM, MAX_NODES
from ValueNetworkFlat import FLAT_CONFIGS, build_flat_model, count_parameters
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
                             "wide = 22.57x capacity (1,649,409). Run both.")
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

    # Built once here only to report the count before the run starts; the real
    # model is constructed by the factory inside train_model, AFTER seeding, so
    # this throwaway must not touch the global RNG order that matters. It does
    # draw from the torch RNG, but train_model calls seed_everything before
    # building anything, which resets it.
    _, probe = build_flat_model(args.arch)
    n_params = count_parameters(probe)

    print(f"=== FLAT-MLP ABLATION ({args.arch}) ===")
    print(f"  input      : {FLAT_DIM:,} ({MAX_NODES} nodes x 99 + 19 global)")
    print(f"  shape      : {shape}")
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
        },
    )


if __name__ == "__main__":
    main()
