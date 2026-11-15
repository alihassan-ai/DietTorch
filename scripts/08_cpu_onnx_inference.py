"""
Deliverable 8: ONNX CPU Inference and Benchmarking Pipeline
Executes inference on exported ONNX models using ONNX Runtime CPUExecutionProvider.
Measures:
- Latency (batch & per-sample ms)
- Inference Throughput (samples/sec)
- Prediction accuracy / agreement with test targets
"""

import os
import sys
import argparse
import time
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import get_cifar10_loaders, CIFAR10_CLASSES


def benchmark_onnx_session(session, input_tensor, num_warmup=10, num_runs=50):
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: input_tensor}

    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, ort_inputs)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, ort_inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    batch_size = input_tensor.shape[0]
    mean_batch_lat = float(np.mean(latencies))
    lat_per_sample = mean_batch_lat / batch_size
    throughput = (1000.0 / mean_batch_lat) * batch_size

    return {
        "mean_batch_ms": mean_batch_lat,
        "lat_per_sample_ms": lat_per_sample,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime CPU Inference")
    parser.add_argument("--onnx_dir", type=str, default="./onnx_models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=40)
    args = parser.parse_args()

    print("=== [Deliverable 8] ONNX Runtime CPU Inference Benchmark ===")
    _, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    images, targets = next(iter(test_loader))
    input_np = images[:args.batch_size].numpy()
    target_np = targets[:args.batch_size].numpy()

    onnx_files = [
        ("baseline_fp32.onnx", "Baseline FP32 ONNX"),
        ("unstructured_pruned.onnx", "Unstructured Pruned ONNX"),
        ("structured_pruned.onnx", "Structured Pruned ONNX"),
        ("ptq_int8.onnx", "PTQ INT8 Quantized ONNX"),
    ]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    print(f"Benchmarking on {args.batch_size} samples across ONNX models:\n")

    for fname, display_name in onnx_files:
        fpath = os.path.join(args.onnx_dir, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {fname} (not found)")
            continue

        session = ort.InferenceSession(fpath, sess_options=opts, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        raw_out = session.run(None, {input_name: input_np})[0]
        preds = np.argmax(raw_out, axis=1)
        acc = 100.0 * np.sum(preds == target_np) / args.batch_size

        metrics = benchmark_onnx_session(session, input_np, num_warmup=10, num_runs=args.runs)
        print(f"--- {display_name} ({fname}) ---")
        print(f"  Accuracy on batch:    {acc:.1f}%")
        print(f"  Batch Latency:        {metrics['mean_batch_ms']:.2f} ms")
        print(f"  Latency per Sample:   {metrics['lat_per_sample_ms']:.3f} ms")
        print(f"  Throughput:           {metrics['throughput']:.1f} samples/sec")
        print()


if __name__ == "__main__":
    main()
