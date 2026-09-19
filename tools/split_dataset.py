"""
Game-aware train/val split for a tools/generate_data.py output directory.

States from one game must never appear in both splits -- a game-level train/val
leak would let the model partially memorize that specific game's trajectory
rather than genuinely generalizing, and the leak wouldn't show up as an
obviously-too-good val loss the way a naive record-level split's leak might
(consecutive states within a game are highly correlated, so even a few leaked
games can meaningfully inflate val metrics without looking anomalous).

Splits by game_id, not by record: pass 1 streams every shard to collect each
game_id's record count, partitions the SET of game_ids 90/10, then pass 2
re-streams every shard writing each record verbatim into whichever of
train/*.jsonl.gz or val/*.jsonl.gz matches its game_id -- one output shard per
input shard per split, so record order/content within a shard is untouched,
just partitioned. A third, independent pass re-reads the WRITTEN output and
asserts zero game_id overlap, rather than trusting the in-memory split.

Usage:
    python tools/split_dataset.py /tmp/gen50 /tmp/gen50_split
    python tools/split_dataset.py /tmp/gen50 /tmp/gen50_split --val-fraction 0.1 --seed 0
"""
import argparse
import glob
import gzip
import json
import os
import random
import sys


def find_shards(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.jsonl.gz"), recursive=True))


def collect_game_ids(directory):
    ids = set()
    for shard in find_shards(directory):
        with gzip.open(shard, "rt") as f:
            for line in f:
                ids.add(json.loads(line)["game_id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", help="Directory of raw shards (generate_data.py's --out-dir)")
    parser.add_argument("out_dir", help="Output directory; train/ and val/ subdirectories are created here")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    shards = find_shards(args.data_dir)
    if not shards:
        sys.exit(f"ERROR: no shards found under {args.data_dir}")

    print(f"Pass 1/3: scanning {len(shards)} shard(s) under {args.data_dir} for game_id...", file=sys.stderr)
    game_record_counts = {}
    total_records = 0
    for shard in shards:
        with gzip.open(shard, "rt") as f:
            for line in f:
                gid = json.loads(line)["game_id"]
                game_record_counts[gid] = game_record_counts.get(gid, 0) + 1
                total_records += 1

    game_ids = sorted(game_record_counts)
    rng = random.Random(args.seed)
    rng.shuffle(game_ids)

    n_val_games = max(1, round(len(game_ids) * args.val_fraction))
    val_game_ids = set(game_ids[:n_val_games])
    train_game_ids = set(game_ids[n_val_games:])
    assert train_game_ids.isdisjoint(val_game_ids), "BUG: train/val game_id partition overlaps"

    train_records = sum(game_record_counts[g] for g in train_game_ids)
    val_records = sum(game_record_counts[g] for g in val_game_ids)

    print(f"Total: {len(game_ids)} games, {total_records} records")
    print(f"Train: {len(train_game_ids)} games, {train_records} records")
    print(f"Val:   {len(val_game_ids)} games, {val_records} records")

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"Pass 2/3: writing split shards to {train_dir} and {val_dir}...", file=sys.stderr)
    written_train_records = 0
    written_val_records = 0
    for shard in shards:
        rel = os.path.relpath(shard, args.data_dir)
        # Flatten the relative path into one filename component so
        # job_0000/x.jsonl.gz and job_0001/x.jsonl.gz (identical basenames
        # across jobs) can't collide once written into one flat directory.
        flat_name = rel.replace(os.sep, "__")

        train_lines = []
        val_lines = []
        with gzip.open(shard, "rt") as f:
            for line in f:
                gid = json.loads(line)["game_id"]
                (train_lines if gid in train_game_ids else val_lines).append(line)

        if train_lines:
            with gzip.open(os.path.join(train_dir, flat_name), "wt") as out:
                out.writelines(train_lines)
            written_train_records += len(train_lines)
        if val_lines:
            with gzip.open(os.path.join(val_dir, flat_name), "wt") as out:
                out.writelines(val_lines)
            written_val_records += len(val_lines)

    assert written_train_records == train_records, \
        f"BUG: wrote {written_train_records} train records, expected {train_records}"
    assert written_val_records == val_records, \
        f"BUG: wrote {written_val_records} val records, expected {val_records}"

    print(f"Pass 3/3: verifying zero game_id overlap by re-reading the written output...", file=sys.stderr)
    final_train_ids = collect_game_ids(train_dir)
    final_val_ids = collect_game_ids(val_dir)
    overlap = final_train_ids & final_val_ids
    assert not overlap, f"BUG: {len(overlap)} game_id(s) appear in both train/ and val/: {sorted(overlap)[:10]}"

    print(f"Verified: {len(final_train_ids)} train game_ids, {len(final_val_ids)} val game_ids, 0 overlap.")
    print("Done.")


if __name__ == "__main__":
    main()
