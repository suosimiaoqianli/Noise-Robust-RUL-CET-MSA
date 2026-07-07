import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.cet_msa import CETMSARULPredictor


@torch.inference_mode()
def benchmark_cet_msa_inference(model, batch_size, window_length, sensor_channels, feature_dim, device, iterations, use_amp):
    sequence = torch.randn(batch_size, window_length, sensor_channels, device=device)
    handcrafted_features = torch.randn(batch_size, feature_dim, device=device)
    warmup = min(50, max(10, iterations // 4))
    amp_context = torch.autocast(device_type='cuda', dtype=torch.float16) if (use_amp and device.type == 'cuda') else nullcontext()

    for _ in range(warmup):
        with amp_context:
            _ = model(sequence, handcrafted_features)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        starter, ender = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        torch.cuda.synchronize()
        starter.record()
        with amp_context:
            for _ in range(iterations):
                _ = model(sequence, handcrafted_features)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        start_time = time.perf_counter()
        for _ in range(iterations):
            _ = model(sequence, handcrafted_features)
        total_ms = (time.perf_counter() - start_time) * 1000.0
        peak_memory_mb = None

    latency_ms_per_batch = total_ms / iterations
    throughput_samples_per_second = (batch_size * iterations) / (total_ms / 1000.0)
    return latency_ms_per_batch, throughput_samples_per_second, peak_memory_mb


def main():
    parser = argparse.ArgumentParser(description='CET-MSA inference micro-benchmark')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--precision', default='fp16', choices=['fp16', 'fp32', 'both'])
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--batch-sizes', type=str, default='1,64,128')
    parser.add_argument('--window-length', type=int, default=30)
    parser.add_argument('--sensor-channels', type=int, default=17)
    parser.add_argument('--feature-dim', type=int, default=34)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model = CETMSARULPredictor(input_dim=args.sensor_channels, window_length=args.window_length).to(device).eval()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'[Info] Loaded checkpoint: {args.checkpoint}')

    batch_sizes = [int(value.strip()) for value in args.batch_sizes.split(',') if value.strip()]
    precisions = ['fp16', 'fp32'] if args.precision == 'both' else [args.precision]

    print('\n=== CET-MSA Inference Benchmark ===')
    header = f"{'Precision':>9} | {'Batch':>5} | {'Latency ms/batch':>16} | {'Throughput samp/s':>19} | {'Peak Mem MB':>12}"
    print(header)
    print('-' * len(header))

    for precision in precisions:
        use_amp = precision == 'fp16'
        for batch_size in batch_sizes:
            latency_ms, throughput, peak_memory_mb = benchmark_cet_msa_inference(
                model=model,
                batch_size=batch_size,
                window_length=args.window_length,
                sensor_channels=args.sensor_channels,
                feature_dim=args.feature_dim,
                device=device,
                iterations=args.iters,
                use_amp=use_amp,
            )
            peak_memory = f'{peak_memory_mb:,.1f}' if peak_memory_mb is not None else '-'
            print(f'{precision:>9} | {batch_size:>5d} | {latency_ms:>16.3f} | {throughput:>19.1f} | {peak_memory:>12}')


if __name__ == '__main__':
    main()
