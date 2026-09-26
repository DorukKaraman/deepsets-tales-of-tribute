import torch
import gzip
import glob
import json
import os
import random
from torch.utils.data import IterableDataset, get_worker_info
from StateParser import json_to_pyg_graph

DEFAULT_SHUFFLE_BUFFER_SIZE = 100_000


class SakkirinaStreamDataset(IterableDataset):
    """Streams (state, outcome) pairs from a directory of gzipped JSONL
    shards, as produced by tools/generate_data.py (<data_dir>/job_*/*.jsonl.gz)
    or tools/split_dataset.py (<data_dir>/{train,val}/*.jsonl.gz) -- any
    directory tree, globbed recursively for *.jsonl.gz.

    Consecutive records within one shard come from the same game and share
    the same outcome label, so without shuffling, a batch_size-256 batch
    drawn straight off the stream can be near single-class. shuffle_buffer_size
    controls a reservoir-style shuffle buffer (larger = better mixing, more
    memory) that fixes this without loading the whole dataset into memory.
    """
    def __init__(self, data_dir, shuffle_buffer_size=DEFAULT_SHUFFLE_BUFFER_SIZE):
        self.data_dir = data_dir
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shards = sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl.gz"), recursive=True))
        if not self.shards:
            raise FileNotFoundError(f"No *.jsonl.gz shards found under {data_dir}")

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # PARTITION FIRST, THEN SHUFFLE. The order matters and getting it
        # backwards silently corrupts the epoch.
        #
        # This used to shuffle the full list and then slice it:
        #
        #     shards = list(self.shards)
        #     random.shuffle(shards)                       # WRONG
        #     worker_shards = shards[worker_id::num_workers]
        #
        # Each worker is a separate process with its own `random` state, seeded
        # per worker (see train_local.worker_init_fn), so every worker shuffled
        # into a DIFFERENT permutation and then took its own stride from it.
        # Slices of different permutations are not a partition: some shards were
        # read by several workers, others by none. Measured over the 128-shard
        # training corpus, per epoch, averaged over 400 runs:
        #
        #     workers   missed/epoch   duplicated/epoch   never in 3 epochs
        #        4       31.7% ±2.6       26.2% ±2.3          3.1% ±1.4
        #        7       33.9% ±2.6       26.3% ±2.1          4.0% ±1.5
        #
        # A shard was picked by each worker independently with probability
        # 1/nw, so the miss rate is (1-1/nw)^nw -- 0.3164 at four workers,
        # 0.3399 at seven, tending to 1/e. The measurement lands on those
        # values, which is what confirms the mechanism rather than merely the
        # symptom. tools/test_stream_dataset_sharding.py is the regression test.
        #
        # At num_workers=0 or 1 this fix changes NOTHING, bit for bit: both
        # orderings reduce to shuffling the whole list with the same RNG state,
        # so they produce the identical permutation. That is why the flat-MLP
        # ablation (REPRODUCE.md section 9), which was run single-process
        # precisely to sidestep this bug, did not need re-running afterwards.
        #
        # Slicing self.shards (already sorted, and identical in every worker)
        # gives a real partition; shuffling afterwards keeps the fresh per-epoch
        # order that the shuffle was there for, since __iter__ runs once per
        # epoch. Each worker still only opens the files it will actually read.
        worker_shards = self.shards[worker_id::num_workers]
        random.shuffle(worker_shards)

        print(f"Worker {worker_id}/{num_workers}: streaming {len(worker_shards)}/{len(self.shards)} shard(s) "
              f"from {self.data_dir} (shuffle_buffer_size={self.shuffle_buffer_size})")

        buffer = []
        error_count = 0
        first_error = None

        for shard_path in worker_shards:
            with gzip.open(shard_path, 'rt') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        outcome = row['outcome']
                        y = torch.tensor([[float(outcome)]], dtype=torch.float32)
                        raw_state = row['data']['state']

                        graph_data = json_to_pyg_graph(raw_state)
                        graph_data.y = y

                        if len(buffer) < self.shuffle_buffer_size:
                            buffer.append(graph_data)
                        else:
                            idx = random.randint(0, self.shuffle_buffer_size - 1)
                            yield buffer[idx]
                            buffer[idx] = graph_data
                    except Exception as e:
                        # A single malformed line must not kill a multi-hour
                        # training run, but silently swallowing everything
                        # (the old behavior) can hide a systematic parsing
                        # bug -- surface the count and the first cause.
                        error_count += 1
                        if first_error is None:
                            first_error = f"{shard_path}: {e!r}"

        if error_count:
            print(f"Worker {worker_id}: WARNING skipped {error_count} malformed record(s); first cause: {first_error}")

        random.shuffle(buffer)
        for data in buffer:
            yield data
