"""
Check that SakkirinaStreamDataset reads every shard exactly once per epoch, at
every worker count.

Partitioning has to happen before shuffling. Each DataLoader worker has its own
RNG, so shuffling the full list and then slicing gives each worker a different
permutation, and the slices stop being a partition. That only shows with two or
more workers, so the test runs 1, 2, 4 and 7, and confirms that shuffle-then-
slice does fail, so that the test can catch it.

Uses small synthetic shards, since only shard selection matters.

    python tools/test_stream_dataset_sharding.py

Exits nonzero on any failure. Writes only to a temp directory it removes.
"""
import gzip
import json
import os
import shutil
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))

try:
    import torch  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402
    from stream_dataset import SakkirinaStreamDataset  # noqa: E402
except ImportError as e:  # pragma: no cover - environment problem, not logic
    sys.exit(f"ERROR: could not import the training modules ({e}).\n"
             f"       Activate the venv built by scripts/setup_python_env.sh first.")

WORKER_COUNTS = [1, 2, 4, 7]
N_SHARDS = 13          # coprime with most worker counts
RECORDS_PER_SHARD = 5


def make_corpus(root):
    """One directory of tiny gzipped shards. Each record carries a unique id so
    duplicates and omissions are both detectable by counting."""
    total = 0
    for s in range(N_SHARDS):
        path = os.path.join(root, f"shard_{s:03d}.jsonl.gz")
        with gzip.open(path, "wt") as f:
            for r in range(RECORDS_PER_SHARD):
                # The minimum structure SakkirinaStreamDataset will parse: an
                # outcome and a state with at least one card list.
                f.write(json.dumps({
                    "game_id": f"g{s}",
                    "outcome": s % 2,
                    "uid": f"{s}:{r}",
                    "data": {"state": {"TavernAvailableCards": [
                        {"Deck": 0, "Cost": 1, "Type": 0, "HP": -1,
                         "CommonId": None, "Taunt": False}]}},
                }) + "\n")
                total += 1
    return total


class ShardNameDataset(SakkirinaStreamDataset):
    """Yields the shard each worker selects, rather than its records.

    Implements the selection both ways: "old" shuffles then slices, "new" slices
    then shuffles. stream_records below exercises the real class.
    """

    def __init__(self, data_dir, strategy):
        super().__init__(data_dir)
        self.strategy = strategy

    def __iter__(self):
        import random
        from torch.utils.data import get_worker_info
        info = get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1
        if self.strategy == "old":
            shards = list(self.shards)
            random.shuffle(shards)
            worker_shards = shards[wid::nw]
        else:
            worker_shards = self.shards[wid::nw]
            random.shuffle(worker_shards)
        for s in worker_shards:
            yield os.path.basename(s)


def worker_init_fn(worker_id):
    """Matches train_local.worker_init_fn: seeds each worker's `random` from its
    torch seed. Without per-worker seeds the shuffle-then-slice failure would not
    reproduce."""
    import random
    import numpy as np
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def shard_reads(root, strategy, num_workers):
    ds = ShardNameDataset(root, strategy)
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers,
                    worker_init_fn=worker_init_fn)
    return Counter(str(name) for name in dl)


def stream_records(root, num_workers):
    """The real dataset, counting what actually comes out."""
    ds = SakkirinaStreamDataset(root, shuffle_buffer_size=4)
    dl = DataLoader(ds, batch_size=None, num_workers=num_workers,
                    worker_init_fn=worker_init_fn)
    return sum(1 for _ in dl)


def check(strategy, root, expected_records, verbose):
    failures = []
    for nw in WORKER_COUNTS:
        torch.manual_seed(0)
        counts = shard_reads(root, strategy, nw)
        missed = [s for s in (os.path.basename(p) for p in sorted(
            SakkirinaStreamDataset(root).shards)) if s not in counts]
        dup = {s: c for s, c in counts.items() if c > 1}

        ok_shards = not missed and not dup
        if not ok_shards:
            failures.append(
                f"num_workers={nw}: {len(missed)} shard(s) never read, "
                f"{len(dup)} read more than once")

        if verbose:
            print(f"    num_workers={nw:>2}: {len(counts)}/{N_SHARDS} distinct, "
                  f"{len(missed)} missed, {len(dup)} duplicated"
                  f"{'' if ok_shards else '   <-- FAIL'}")
    return failures


def check_records(root, expected_records, verbose):
    failures = []
    for nw in WORKER_COUNTS:
        torch.manual_seed(0)
        got = stream_records(root, nw)
        if got != expected_records:
            failures.append(f"num_workers={nw}: streamed {got} records, "
                            f"dataset holds {expected_records}")
        if verbose:
            print(f"    num_workers={nw:>2}: streamed {got}/{expected_records} records"
                  f"{'' if got == expected_records else '   <-- FAIL'}")
    return failures


def main():
    root = tempfile.mkdtemp(prefix="sharding_test_")
    try:
        expected = make_corpus(root)
        print(f"{N_SHARDS} synthetic shards, {expected} records, "
              f"worker counts {WORKER_COUNTS}\n")

        # 1. Shuffle-then-slice must fail, or the test could not catch it.
        print("  OLD strategy (shuffle then slice) -- expected to FAIL:")
        old_failures = check("old", root, expected, verbose=True)
        print()

        # 2. Slice-then-shuffle must cover every shard once on selection...
        print("  CURRENT strategy (slice then shuffle) -- shard coverage:")
        new_failures = check("new", root, expected, verbose=True)
        print()

        # 3. ...and end to end, on the real class, counting records out.
        print("  CURRENT code, real SakkirinaStreamDataset -- record count:")
        record_failures = check_records(root, expected, verbose=True)
        print()

        if not old_failures:
            sys.exit("FAILED: the old shuffle-then-slice strategy did NOT lose or "
                     "duplicate any shard in this run.\n"
                     "        The test is not exercising the bug it exists to catch -- "
                     "check that\n"
                     "        worker_init_fn still gives each worker a distinct "
                     "`random` seed.")
        print(f"  Old strategy failed at {len(old_failures)} worker count(s), as it must:")
        for f in old_failures:
            print(f"    {f}")
        print()

        problems = new_failures + record_failures
        if problems:
            print("FAILED: the current code does not partition shards cleanly.")
            for f in problems:
                print(f"  {f}")
            sys.exit(1)

        print("PASSED: every shard is read exactly once per epoch at every worker "
              "count, and the\n        record count out equals the record count in.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
