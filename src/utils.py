"""
Benchmarking, profiling, and metric utilities for PyTorch models.
Provides exact calculations for model size, parameter count, sparsity, latency, throughput, and accuracy.
"""

import os
import time
import torch
import pandas as pd
import numpy as np


def count_parameters(model):
    """
    Counts total parameters and non-zero parameters to measure sparsity.
    """
    total_params = 0
    non_zero_params = 0

    if isinstance(model, torch.jit.ScriptModule):
        # Scripted quantized models
        for p in model.parameters():
            num = p.numel()
            total_params += num
            non_zero_params += torch.count_nonzero(p).item()
    else:
        for p in model.parameters():
            num = p.numel()
            total_params += num
            non_zero_params += torch.count_nonzero(p).item()

    sparsity_pct = 100.0 * (1.0 - (non_zero_params / max(1, total_params)))
    return total_params, non_zero_params, sparsity_pct


def get_model_size_mb(filepath):
    """Returns file size of a saved model checkpoint in Megabytes (MB)."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0


def evaluate_model(model, data_loader, device="cpu", max_batches=None):
    """
    Evaluates top-1 classification accuracy on a dataset.
    """
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule) and hasattr(model, "to"):
        model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    accuracy = 100.0 * correct / max(1, total)
    return accuracy, correct, total


def benchmark_inference_speed(
    model, sample_input, device="cpu", num_warmup=10, num_runs=50
):
    """
    Accurately benchmarks inference latency and throughput on CPU/device.
    Uses warmup runs to eliminate cold-start transients.
    """
    model.eval()
    if not isinstance(model, torch.jit.ScriptModule) and hasattr(model, "to"):
        model.to(device)
    sample_input = sample_input.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_input)

    # Synchronize if GPU/MPS
    if device == "cuda":
        torch.cuda.synchronize()

    batch_size = sample_input.size(0)
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_input)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_batch_latency_ms = float(np.mean(latencies))
    std_batch_latency_ms = float(np.std(latencies))
    latency_per_sample_ms = mean_batch_latency_ms / batch_size
    throughput_samples_per_sec = (1000.0 / mean_batch_latency_ms) * batch_size

    return {
        "mean_batch_latency_ms": mean_batch_latency_ms,
        "std_batch_latency_ms": std_batch_latency_ms,
        "latency_per_sample_ms": latency_per_sample_ms,
        "throughput_samples_per_sec": throughput_samples_per_sec,
    }


def save_benchmark_csv(records, output_path="./results/benchmark_results.csv"):
    """Saves benchmark comparison dictionary list to a clean CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"[Results] Benchmark CSV saved to {output_path}")
    return df
