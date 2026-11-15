"""
Deliverable 7: Automated Benchmarking and CSV Generation Script
Evaluates all five model variants on:
- Top-1 Accuracy (%) on CIFAR-10 test set
- Disk File Size (MB) and Compression Ratio
- Parameter Counts (Total vs Active Non-Zero) and Sparsity (%)
- CPU Inference Latency (Batch & Sample in ms)
- Throughput (Samples / Second)

Generates:
1. results/benchmark_results.csv
2. results/test_predictions_summary.csv
3. results/accuracy_vs_size.png
4. results/latency_vs_throughput.png
"""

import os
import sys
import argparse
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_model_for_inference, MODEL_PATHS
from src.dataset import get_cifar10_loaders, CIFAR10_CLASSES
from src.utils import evaluate_model, count_parameters, get_model_size_mb, benchmark_inference_speed


def run_benchmark(batch_size=32, num_timing_runs=40, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    print("=== [Deliverable 7] Automated Benchmarking Across All 5 Models ===")

    _, test_loader = get_cifar10_loaders(batch_size=batch_size)

    sample_images, sample_targets = next(iter(test_loader))
    sample_images = sample_images[:batch_size]

    models = [
        ("Baseline FP32", "baseline_fp32", MODEL_PATHS["baseline_fp32"]),
        ("PTQ INT8", "ptq_int8", MODEL_PATHS["ptq_int8"]),
        ("QAT INT8", "qat_int8", MODEL_PATHS["qat_int8"]),
        ("Unstructured Pruned", "unstructured_pruned", MODEL_PATHS["unstructured_pruned"]),
        ("Structured Pruned", "structured_pruned", MODEL_PATHS["structured_pruned"]),
    ]

    records = []
    predictions_records = []

    fp32_size = get_model_size_mb(MODEL_PATHS["baseline_fp32"])

    for display_name, key, path in models:
        print(f"\nBenchmarking {display_name} ({path})...")
        model = load_model_for_inference(key, path)

        # 1. Model Size & Parameters
        size_mb = get_model_size_mb(path)
        comp_ratio = fp32_size / max(0.001, size_mb)
        total_p, non_zero_p, sparsity = count_parameters(model)

        # 2. Accuracy Evaluation (on 2000 test images for rapid benchmark)
        print("  Evaluating test accuracy...")
        accuracy, correct, total = evaluate_model(model, test_loader, device="cpu", max_batches=60)
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")

        # 3. Latency & Throughput Benchmark
        print("  Benchmarking CPU latency and throughput...")
        timing = benchmark_inference_speed(
            model, sample_images, device="cpu", num_warmup=10, num_runs=num_timing_runs
        )
        print(f"  Latency/Sample: {timing['latency_per_sample_ms']:.2f} ms | Throughput: {timing['throughput_samples_per_sec']:.1f} samples/s")

        records.append({
            "Model Variant": display_name,
            "Precision": "INT8" if "INT8" in display_name else "FP32",
            "Test Accuracy (%)": round(accuracy, 2),
            "Model Size (MB)": round(size_mb, 2),
            "Compression Ratio": round(comp_ratio, 2),
            "Total Parameters": total_p,
            "Active Parameters": non_zero_p,
            "Sparsity (%)": round(sparsity, 2),
            "Batch Latency (ms)": round(timing["mean_batch_latency_ms"], 2),
            "Latency per Sample (ms)": round(timing["latency_per_sample_ms"], 3),
            "Throughput (samples/s)": round(timing["throughput_samples_per_sec"], 1),
        })

        # Collect sample predictions for first 10 items
        with torch.no_grad():
            sample_logits = model(sample_images[:10])
            sample_preds = sample_logits.argmax(dim=1)
            for idx in range(10):
                predictions_records.append({
                    "Model": display_name,
                    "Sample Index": idx,
                    "True Label": CIFAR10_CLASSES[sample_targets[idx].item()],
                    "Predicted Label": CIFAR10_CLASSES[sample_preds[idx].item()],
                    "Correct": bool(sample_targets[idx] == sample_preds[idx]),
                })

    # Save CSVs
    results_df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[Done] Benchmark results saved to {csv_path}")
    print(results_df.to_string(index=False))

    preds_df = pd.DataFrame(predictions_records)
    preds_csv_path = os.path.join(output_dir, "test_predictions_summary.csv")
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[Done] Test predictions summary saved to {preds_csv_path}")

    # Generate Visual Comparison Plots
    generate_benchmark_plots(results_df, output_dir)
    return results_df


def generate_benchmark_plots(df, output_dir):
    """Generates comparison bar charts and scatter plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Model Size vs Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, row in df.iterrows():
        ax.scatter(row["Model Size (MB)"], row["Test Accuracy (%)"], color=colors[i % len(colors)], s=250, label=row["Model Variant"], edgecolors="black", linewidth=1.5)
        ax.annotate(
            row["Model Variant"],
            (row["Model Size (MB)"], row["Test Accuracy (%)"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Edge Model Optimization: Accuracy vs. Model Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model Size (MB) [Lower is Better]", fontsize=11)
    ax.set_ylabel("Test Accuracy (%) [Higher is Better]", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "accuracy_vs_size.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved Accuracy vs Size plot to {plot1_path}")

    # Plot 2: Latency vs Throughput
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = df["Model Variant"].tolist()
    y_pos = np.arange(len(names))

    # Latency Bar Chart
    ax1.barh(y_pos, df["Latency per Sample (ms)"], color="#3498db", edgecolor="#2980b9")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Latency per Sample (ms) [Lower is Better]", fontsize=10)
    ax1.set_title("CPU Inference Latency per Sample", fontsize=11, fontweight="bold")
    for i, v in enumerate(df["Latency per Sample (ms)"]):
        ax1.text(v + 0.1, i, f"{v:.2f} ms", va="center", fontsize=9, fontweight="bold")

    # Throughput Bar Chart
    ax2.barh(y_pos, df["Throughput (samples/s)"], color="#2ecc71", edgecolor="#27ae60")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Throughput (samples/sec) [Higher is Better]", fontsize=10)
    ax2.set_title("Inference Throughput on CPU", fontsize=11, fontweight="bold")
    for i, v in enumerate(df["Throughput (samples/s)"]):
        ax2.text(v + 2, i, f"{v:.1f}/s", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "latency_vs_throughput.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved Latency vs Throughput plot to {plot2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full automated benchmark across 5 models")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    run_benchmark(batch_size=args.batch_size, num_timing_runs=args.runs, output_dir=args.output_dir)
