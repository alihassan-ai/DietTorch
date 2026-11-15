"""
Master Reproducible Execution Pipeline
Executes the complete optimization pipeline from Deliverable 1 to Deliverable 9:
1. Baseline FP32 Model Initialization / Evaluation
2. Post-Training Static Quantization (PTQ-INT8)
3. Quantization-Aware Training (QAT-INT8)
4. Iterative Unstructured Pruning (50% Sparsity)
5. Structured Channel Pruning & Architecture Reconstruction
6. Unified Inference Verification
7. Automated Benchmarking & CSV Generation
8. ONNX Export and CPU Inference Validation
9. Publication-Quality PDF Technical Report Generation
"""

import os
import sys
import time
import subprocess
import argparse


SCRIPTS = [
    ("Deliverable 1: Baseline FP32 Model", "scripts/01_train_baseline.py"),
    ("Deliverable 2: Post-Training Static Quantization (PTQ)", "scripts/02_ptq_quantization.py"),
    ("Deliverable 3: Quantization-Aware Training (QAT)", "scripts/03_qat_quantization.py"),
    ("Deliverable 4: Unstructured Magnitude Pruning", "scripts/04_unstructured_pruning.py"),
    ("Deliverable 5: Structured Channel Pruning & Reconstruction", "scripts/05_structured_pruning.py"),
    ("Deliverable 6: Unified Inference Pipeline", "scripts/06_inference.py"),
    ("Deliverable 7: Automated Benchmarking & CSV Generation", "scripts/07_benchmark.py"),
    ("Deliverable 8: ONNX Export", "scripts/08_export_onnx.py"),
    ("Deliverable 8: ONNX CPU Inference", "scripts/08_cpu_onnx_inference.py"),
    ("Deliverable 9: Technical Report PDF Generator", "scripts/09_generate_report.py"),
]


def run_command(cmd, desc):
    print(f"\n{'='*70}")
    print(f"--> [RUNNING] {desc}")
    print(f"    Command: {cmd}")
    print(f"{'='*70}\n")
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd.split()[1:], check=True)
    dt = time.time() - t0
    print(f"\n--> [COMPLETED] {desc} in {dt:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete optimization and benchmarking pipeline")
    parser.add_argument("--skip_training", action="store_true", help="Skip training / pruning fine-tuning if checkpoints already exist")
    args = parser.parse_args()

    start_total = time.time()
    print("======================================================================")
    print("  EDGE COMPUTING & INTELLIGENT SYSTEMS — ASSIGNMENT # 03 PIPELINE")
    print("======================================================================")

    for desc, script_path in SCRIPTS:
        if args.skip_training and script_path in [
            "scripts/01_train_baseline.py",
            "scripts/02_ptq_quantization.py",
            "scripts/03_qat_quantization.py",
            "scripts/04_unstructured_pruning.py",
            "scripts/05_structured_pruning.py",
        ]:
            print(f"Skipping {desc} (--skip_training enabled)")
            continue

        cmd = f"python3 {script_path}"
        run_command(cmd, desc)

    total_time = time.time() - start_total
    print("\n" + "="*70)
    print(f"SUCCESS: Complete Pipeline Finished in {total_time:.1f}s ({total_time/60:.1f} min)")
    print("All checkpoints, CSV results, plots, ONNX models, and PDF report ready!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
