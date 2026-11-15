"""
Submission Packaging Script for Edge Computing Assignment 3
Bundles all 10 required deliverables into ROLL-NUM_SECTION.zip according to the course guidelines.
Excludes raw dataset directories and temporary cache files.
"""

import os
import sys
import zipfile
import argparse


DELIVERABLE_PATHS = [
    # Deliverable 10: README
    "README.md",
    "run_all.py",
    # Deliverable 9: Technical Documentation PDF
    "technical_report.pdf",
    "documentation.pdf",
    # Source modules
    "src/__init__.py",
    "src/model.py",
    "src/dataset.py",
    "src/utils.py",
    # Deliverables 1-8 Scripts
    "scripts/01_train_baseline.py",
    "scripts/02_ptq_quantization.py",
    "scripts/03_qat_quantization.py",
    "scripts/04_unstructured_pruning.py",
    "scripts/05_structured_pruning.py",
    "scripts/06_inference.py",
    "scripts/07_benchmark.py",
    "scripts/08_export_onnx.py",
    "scripts/08_cpu_onnx_inference.py",
    "scripts/09_generate_report.py",
    # Deliverable 1-5 Checkpoints
    "checkpoints/baseline_fp32.pt",
    "checkpoints/ptq_int8.pt",
    "checkpoints/qat_int8.pt",
    "checkpoints/unstructured_pruned.pt",
    "checkpoints/structured_pruned.pt",
    "checkpoints/structured_pruned_state_dict.pt",
    # Deliverable 7 Results
    "results/benchmark_results.csv",
    "results/test_predictions_summary.csv",
    "results/accuracy_vs_size.png",
    "results/latency_vs_throughput.png",
    # Deliverable 8 Exported ONNX Models
    "onnx_models/baseline_fp32.onnx",
    "onnx_models/unstructured_pruned.onnx",
    "onnx_models/structured_pruned.onnx",
    "onnx_models/ptq_int8.onnx",
]


def create_submission_zip(roll_num="22i-0001_A", output_dir="."):
    zip_name = f"{roll_num}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    print(f"=== Creating Submission Archive: {zip_path} ===")
    missing_files = []
    included_files = []

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for rel_path in DELIVERABLE_PATHS:
            if os.path.exists(rel_path):
                # Archive with folder prefix
                arcname = os.path.join(roll_num, rel_path)
                zipf.write(rel_path, arcname)
                included_files.append(rel_path)
            else:
                missing_files.append(rel_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[Success] Submission zip created: {zip_path} ({zip_size_mb:.2f} MB)")
    print(f"Total files included: {len(included_files)}")
    if missing_files:
        print(f"Warning: {len(missing_files)} optional files were not found: {missing_files}")

    return zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Assignment 3 submission")
    parser.add_argument("--roll_num", type=str, default="22i-0001_A", help="Student Roll Number and Section (e.g. 22i-0001_A)")
    args = parser.parse_args()

    create_submission_zip(roll_num=args.roll_num)
