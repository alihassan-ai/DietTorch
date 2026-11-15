# ⚡ DietTorch: Deep Model Compression, Quantization & Pruning Toolkit
> **Put your PyTorch models on a diet — 4x smaller, 3x faster on edge CPUs & NPUs.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-blue.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Edge Computing & Intelligent Systems (Assignment #03 | Roll No: 22i-0541)*

---

## 📌 Project Overview
This repository delivers an end-to-end edge AI optimization pipeline developed in PyTorch and ONNX Runtime. The project evaluates model compression techniques across five executable model variants trained and benchmarked on the **CIFAR-10** image classification benchmark:

1. **Baseline Model (FP32)**: Adapted CIFAR-10 ResNet-18 architecture.
2. **Post-Training Static Quantization (PTQ-INT8)**: Fused Conv-BN-ReLU layers, calibrated activation ranges, and converted to INT8.
3. **Quantization-Aware Training (QAT-INT8)**: Inserted fake-quantization modules with backpropagation and Straight-Through Estimators (STE).
4. **Iterative Unstructured Pruning**: Magnitude-based L1 pruning across Conv2d and Linear layers achieving 50% parameter sparsity.
5. **Structured Channel Pruning & Physical Reconstruction**: L1-norm filter importance ranking with true physical architecture reconstruction (29.56% physical parameter reduction).

---

## 📋 Deliverables Matrix (Requirements 1 – 10)

| # | Deliverable | Description | Key Implementation Files / Artifacts |
|---|-------------|-------------|---------------------------------------|
| **1** | **Baseline Model** | Trained FP32 CIFAR-10 ResNet-18 checkpoint and script | [`scripts/01_train_baseline.py`](scripts/01_train_baseline.py), [`checkpoints/baseline_fp32.pt`](checkpoints/baseline_fp32.pt) |
| **2** | **PTQ Implementation** | Model fusion, calibration pipeline, INT8 conversion | [`scripts/02_ptq_quantization.py`](scripts/02_ptq_quantization.py), [`checkpoints/ptq_int8.pt`](checkpoints/ptq_int8.pt) |
| **3** | **QAT Implementation** | Fake quantization insertion, fine-tuning, INT8 conversion | [`scripts/03_qat_quantization.py`](scripts/03_qat_quantization.py), [`checkpoints/qat_int8.pt`](checkpoints/qat_int8.pt) |
| **4** | **Unstructured Pruning** | Magnitude-based iterative pruning (50% sparsity) & permanent masks | [`scripts/04_unstructured_pruning.py`](scripts/04_unstructured_pruning.py), [`checkpoints/unstructured_pruned.pt`](checkpoints/unstructured_pruned.pt) |
| **5** | **Structured Pruning** | Channel pruning & physical architecture reconstruction | [`scripts/05_structured_pruning.py`](scripts/05_structured_pruning.py), [`checkpoints/structured_pruned.pt`](checkpoints/structured_pruned.pt) |
| **6** | **Inference Pipeline** | Unified executable inference pipeline supporting all five models | [`scripts/06_inference.py`](scripts/06_inference.py) |
| **7** | **Automated Benchmarking** | Comprehensive benchmarking script, metrics CSV, and visual plots | [`scripts/07_benchmark.py`](scripts/07_benchmark.py), [`results/benchmark_results.csv`](results/benchmark_results.csv) |
| **8** | **ONNX Deployment** | Exported ONNX models and ONNX Runtime CPU inference benchmarking | [`scripts/08_export_onnx.py`](scripts/08_export_onnx.py), [`scripts/08_cpu_onnx_inference.py`](scripts/08_cpu_onnx_inference.py), [`onnx_models/`](onnx_models/) |
| **9** | **Technical Report** | Publication-quality PDF documentation with theory, results, and charts | [`technical_report.pdf`](technical_report.pdf) / [`documentation.pdf`](documentation.pdf) |
| **10** | **README** | Setup instructions, directory map, and execution guidelines | [`README.md`](README.md) |

---

## 📊 Experimental Benchmarking Results

Evaluated on CIFAR-10 test set using CPU inference (batch size 32, 40 benchmark trials with 10 warmup runs):

| Model Variant | Precision | Test Accuracy (%) | Disk Size (MB) | Compression Ratio | Active Parameters | Sparsity (%) | CPU Latency / Sample | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet-18** | FP32 | **95.10%** | 42.70 MB | 1.00x | 11,173,962 | 0.00% | 5.18 ms | 192.9 |
| **PTQ Quantized** | INT8 | **94.58%** | **10.81 MB** | **3.95x** | 11,173,962 (int8) | 0.00% | 1.88 ms | 532.2 |
| **QAT Quantized** | INT8 | **94.90%** | **10.81 MB** | **3.95x** | 11,173,962 (int8) | 0.00% | **1.83 ms** | **547.4** |
| **Unstructured Pruned** | FP32 | **94.64%** | 42.70 MB | 1.00x | 5,591,786 | **49.96%** | 5.23 ms | 191.0 |
| **Structured Pruned** | FP32 | **93.39%** | **30.11 MB** | **1.42x** | **7,871,178** | 0.00% | 4.84 ms | 206.7 |

### Key Experimental Insights:
- **QAT (Quantization-Aware Training)** achieves the best speedup and memory efficiency: **3.95x compression** and **2.83x CPU throughput acceleration** with only a 0.20% delta from the FP32 baseline.
- **Structured Channel Pruning** physically eliminates **29.56% of parameters**, reducing model size from 42.70 MB to 30.11 MB and executing natively on standard CPU/NPU hardware without sparse engine requirements.
- **Unstructured Pruning** zeroes **49.96% of all weights** while preserving 94.87% accuracy, providing optimal compression for over-the-air (OTA) bandwidth transmission.

---

## 🗂️ Project Directory Structure

```text
Assignment 3/
├── checkpoints/                          # Executable PyTorch model checkpoints (.pt)
│   ├── baseline_fp32.pt                  # 1. Baseline FP32 ResNet-18 checkpoint
│   ├── ptq_int8.pt                       # 2. PTQ INT8 TorchScript executable checkpoint
│   ├── qat_int8.pt                       # 3. QAT INT8 TorchScript executable checkpoint
│   ├── unstructured_pruned.pt            # 4. Unstructured Pruned checkpoint (50% sparsity)
│   ├── structured_pruned.pt              # 5. Reconstructed Structured Pruned checkpoint
│   └── structured_pruned_state_dict.pt   # Reconstructed model state_dict
├── onnx_models/                          # Exported ONNX models
│   ├── baseline_fp32.onnx                # Baseline FP32 ONNX model
│   ├── unstructured_pruned.onnx          # Unstructured Pruned ONNX model
│   ├── structured_pruned.onnx            # Reconstructed Structured Pruned ONNX model
│   └── ptq_int8.onnx                     # Quantized INT8 ONNX model
├── results/                              # Generated benchmarking data and visual charts
│   ├── benchmark_results.csv             # Full benchmark comparison metrics across 5 models
│   ├── test_predictions_summary.csv      # Sample inference predictions and confidence scores
│   ├── accuracy_vs_size.png              # Accuracy vs. Model Size trade-off plot
│   └── latency_vs_throughput.png         # CPU Latency & Throughput comparison plot
├── src/                                  # Core reusable library modules
│   ├── __init__.py
│   ├── model.py                          # CIFAR-10 ResNet-18 quantizable & pruning architecture
│   ├── dataset.py                        # CIFAR-10 dataset loaders and calibration subset
│   └── utils.py                          # Profiling, timing, sparsity, and parameter counters
├── scripts/                              # Standalone executable pipeline scripts
│   ├── 01_train_baseline.py              # Deliverable 1: Baseline FP32 training/evaluation
│   ├── 02_ptq_quantization.py            # Deliverable 2: Post-Training Static Quantization (PTQ)
│   ├── 03_qat_quantization.py            # Deliverable 3: Quantization-Aware Training (QAT)
│   ├── 04_unstructured_pruning.py        # Deliverable 4: Magnitude-based iterative pruning
│   ├── 05_structured_pruning.py          # Deliverable 5: Channel pruning & architecture reconstruction
│   ├── 06_inference.py                   # Deliverable 6: Unified inference script for all 5 models
│   ├── 07_benchmark.py                   # Deliverable 7: Automated benchmarking & CSV generation
│   ├── 08_export_onnx.py                 # Deliverable 8: ONNX export script
│   ├── 08_cpu_onnx_inference.py          # Deliverable 8: ONNX Runtime CPU inference benchmark
│   └── 09_generate_report.py             # Deliverable 9: PDF technical report generator
├── run_all.py                            # Master reproducible execution script
├── package_submission.py                 # Packaging utility to create ROLL-NUM_SECTION.zip
├── technical_report.pdf                  # Deliverable 9: Publication-quality documentation PDF
├── documentation.pdf                     # Mirror copy of technical_report.pdf
└── README.md                             # Deliverable 10: Complete technical documentation
```

---

## ⚙️ Installation & Environment Setup

### 1. Requirements
- Python 3.10+ (Tested on Python 3.12)
- PyTorch & Torchvision
- ONNX & ONNX Runtime
- Pandas, NumPy, Matplotlib
- ReportLab (for PDF generation)

### 2. Install Dependencies
```bash
pip install torch torchvision onnx onnxruntime pandas numpy matplotlib reportlab
```

---

## 🚀 Step-by-Step Execution Guide

### Option A: Complete Automated Pipeline (Recommended)
To run the full end-to-end pipeline deterministically in a single command:
```bash
python3 run_all.py
```
To evaluate existing checkpoints without re-running training fine-tuning:
```bash
python3 run_all.py --skip_training
```

---

### Option B: Running Individual Deliverable Modules

#### 1. Baseline FP32 Model (Deliverable 1)
```bash
python3 scripts/01_train_baseline.py
```

#### 2. Post-Training Static Quantization (PTQ) (Deliverable 2)
```bash
python3 scripts/02_ptq_quantization.py --calibration_samples 1000
```

#### 3. Quantization-Aware Training (QAT) (Deliverable 3)
```bash
python3 scripts/03_qat_quantization.py --steps 150 --lr 1e-4
```

#### 4. Iterative Unstructured Pruning (Deliverable 4)
```bash
python3 scripts/04_unstructured_pruning.py --sparsity 0.50 --iterations 3
```

#### 5. Structured Channel Pruning & Reconstruction (Deliverable 5)
```bash
python3 scripts/05_structured_pruning.py --prune_ratio 0.30 --steps 150
```

#### 6. Unified Inference Pipeline (Deliverable 6)
Run inference on test samples across all five models:
```bash
# Test all five models
python3 scripts/06_inference.py --model all --num_samples 10

# Test an individual model variant
python3 scripts/06_inference.py --model qat_int8 --num_samples 10
```

#### 7. Automated Benchmarking & Results CSV (Deliverable 7)
```bash
python3 scripts/07_benchmark.py --batch_size 32 --runs 40
```
Outputs generated:
- `results/benchmark_results.csv`
- `results/test_predictions_summary.csv`
- `results/accuracy_vs_size.png`
- `results/latency_vs_throughput.png`

#### 8. ONNX Export and CPU Deployment (Deliverable 8)
```bash
# Export models to ONNX
python3 scripts/08_export_onnx.py

# Benchmark CPU execution via ONNX Runtime
python3 scripts/08_cpu_onnx_inference.py
```

#### 9. Generate Technical Report PDF (Deliverable 9)
```bash
python3 scripts/09_generate_report.py
```
Generates `technical_report.pdf` and `documentation.pdf`.

---

## 📦 Submission Packaging

To generate the final submission zip formatted according to course guidelines (`ROLL-NUM_SECTION.zip`):
```bash
python3 package_submission.py --roll_num 22i-0001_A
```
*(Replace `22i-0001_A` with your official Roll Number and Section).*
The script automatically packages all 10 required deliverables into a single clean zip archive excluding raw dataset caches.
