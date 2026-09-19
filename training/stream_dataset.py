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

        # Shuffle shard order (fresh each epoch, since __iter__ is called
        # once per epoch) before splitting whole shards across workers --
        # cheaper than the old per-line "i % num_workers" split, since each
        # worker only ever opens the files it's actually going to read.
        shards = list(self.shards)
        random.shuffle(shards)
        worker_shards = shards[worker_id::num_workers]

        print(f"Worker {worker_id}/{num_workers}: streaming {len(worker_shards)}/{len(shards)} shard(s) "
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
