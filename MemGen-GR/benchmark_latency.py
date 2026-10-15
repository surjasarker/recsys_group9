"""
Per-query inference latency benchmark for TIGER and SASRec.

Loads a saved checkpoint and times model.generate() on real test data.
Run on a GPU node (checkpoints must be accessible):

  python benchmark_latency.py --model=TIGER --checkpoint_path=<path> --category=Sports_and_Outdoors
  python benchmark_latency.py --model=SASRec --checkpoint_path=<path> --category=Sports_and_Outdoors
"""
import argparse
import time
import statistics

import torch
from torch.utils.data import DataLoader

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


class LatencyPipeline(Pipeline):
    def run(self):
        pass  # override so Pipeline.run() does nothing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',           type=str, required=True)
    parser.add_argument('--dataset',         type=str, default='AmazonReviews2014')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config',          type=str, default=None)
    parser.add_argument('--n_warmup',        type=int, default=5,
                        help='Batches to discard before timing starts')
    parser.add_argument('--n_repeat',        type=int, default=20,
                        help='Batches to time')
    parser.add_argument('--batch_size',      type=int, default=32,
                        help='Queries per batch')
    parser.add_argument('--n_return_sequences', type=int, default=10,
                        help='Top-k to retrieve (must match eval setting)')
    return parser.parse_known_args()


def time_generate(model, batch, device, n_return_sequences):
    batch = {k: v.to(device) for k, v in batch.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(batch, n_return_sequences=n_return_sequences)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000  # ms


def benchmark(pipeline, args):
    device = pipeline.config['device']
    batch_size = args.batch_size

    loader = DataLoader(
        pipeline.tokenized_datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn['test'],
    )

    pipeline.model = pipeline.model.to(device)
    pipeline.model.eval()
    batches = list(loader)

    # Warmup
    print(f'Warming up ({args.n_warmup} batches)...')
    for i in range(min(args.n_warmup, len(batches))):
        time_generate(pipeline.model, batches[i], device, args.n_return_sequences)

    # Timed runs — cycle through available batches
    print(f'Timing {args.n_repeat} batches (batch_size={batch_size})...')
    batch_times_ms = []
    for i in range(args.n_repeat):
        batch = batches[i % len(batches)]
        ms = time_generate(pipeline.model, batch, device, args.n_return_sequences)
        actual_bs = batch[list(batch.keys())[0]].shape[0]
        batch_times_ms.append(ms / actual_bs)  # per-query ms

    mean_ms  = statistics.mean(batch_times_ms)
    std_ms   = statistics.stdev(batch_times_ms) if len(batch_times_ms) > 1 else 0.0
    med_ms   = statistics.median(batch_times_ms)
    p95_ms   = sorted(batch_times_ms)[int(0.95 * len(batch_times_ms))]

    print()
    print('=' * 60)
    print(f'Model            : {args.model}')
    print(f'Checkpoint       : {args.checkpoint_path.split("/")[-1]}')
    category = pipeline.config.get('category', 'N/A')
    print(f'Category         : {category}')
    print(f'Device           : {device}')
    print(f'Batch size       : {batch_size}')
    print(f'n_return_sequences: {args.n_return_sequences}')
    print(f'Timed batches    : {args.n_repeat}')
    print('-' * 60)
    print(f'Per-query latency (ms):')
    print(f'  Mean  : {mean_ms:.3f} ± {std_ms:.3f}')
    print(f'  Median: {med_ms:.3f}')
    print(f'  P95   : {p95_ms:.3f}')
    print('=' * 60)


def main():
    args, unparsed = parse_args()
    command_line_configs = parse_command_line_args(unparsed)
    command_line_configs.update({'epochs': 0, 'logging': False})

    pipeline = LatencyPipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config,
        config_dict=command_line_configs,
    )

    benchmark(pipeline, args)


if __name__ == '__main__':
    main()
