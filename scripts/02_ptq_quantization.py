"""
Deliverable 2: Post-Training Static Quantization (PTQ) Pipeline
Implements:
1. Model fusion of compatible Conv-BN-ReLU layers.
2. Quantization backend configuration (qnnpack / fbgemm).
3. Calibration pipeline on a representative subset of CIFAR-10 training data.
4. Conversion to an INT8 quantized model.
5. Serialization to an executable TorchScript checkpoint at checkpoints/ptq_int8.pt.
"""

import os
import sys
import argparse
import copy
import torch
import torch.ao.quantization as quantization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18, get_quantization_backend
from src.dataset import get_cifar10_loaders, get_calibration_loader
from src.utils import evaluate_model, count_parameters, get_model_size_mb, benchmark_inference_speed


def run_ptq(
    baseline_checkpoint="./checkpoints/baseline_fp32.pt",
    save_path="./checkpoints/ptq_int8.pt",
    num_calibration_samples=500,
    batch_size=64,
):
    print("=== [Deliverable 2] Post-Training Static Quantization (PTQ) ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Configure Quantization Backend
    backend = get_quantization_backend()
    if backend == "none":
        backend = "qnnpack"
    torch.backends.quantized.engine = backend
    print(f"Quantization Backend Engine: {backend}")

    # 2. Load trained FP32 model
    print(f"Loading baseline FP32 model from {baseline_checkpoint}...")
    model = get_cifar10_resnet18()
    state_dict = torch.load(baseline_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Model Layer Fusion
    print("Applying layer fusion (Conv2d + BatchNorm2d + ReLU)...")
    model.fuse_model(is_qat=False)

    # 4. Set QConfig and Prepare for Static Quantization
    print(f"Configuring qconfig for '{backend}' backend...")
    model.qconfig = quantization.get_default_qconfig(backend)
    prepared_model = quantization.prepare(model, inplace=False)

    # 5. Calibration Pipeline
    print(f"Running calibration pipeline over {num_calibration_samples} training samples...")
    calibration_loader = get_calibration_loader(num_samples=num_calibration_samples, batch_size=batch_size)
    prepared_model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calibration_loader):
            prepared_model(images)
    print("Calibration completed successfully.")

    # 6. Convert to INT8 Quantized Model
    print("Converting calibrated model to INT8 quantized model...")
    quantized_model = quantization.convert(prepared_model, inplace=False)
    quantized_model.eval()

    # 7. Evaluate on CIFAR-10 Test Set
    _, test_loader = get_cifar10_loaders(batch_size=128)
    print("Evaluating PTQ INT8 model on CIFAR-10 test set (CPU)...")
    int8_acc, correct, total = evaluate_model(quantized_model, test_loader, device="cpu")
    print(f"PTQ INT8 Test Accuracy: {int8_acc:.2f}% ({correct}/{total})")

    # 8. Save Executable Model Checkpoint (TorchScript)
    print(f"Exporting executable TorchScript model to {save_path}...")
    sample_input = torch.randn(1, 3, 32, 32)
    scripted_model = torch.jit.trace(quantized_model, sample_input)
    torch.jit.save(scripted_model, save_path)
    print(f"PTQ INT8 checkpoint saved successfully to {save_path}")

    # 9. Verify reload and metrics
    reloaded_model = torch.jit.load(save_path)
    reload_acc, _, _ = evaluate_model(reloaded_model, test_loader, device="cpu")
    print(f"Reloaded Checkpoint Test Accuracy: {reload_acc:.2f}%")

    fp32_size = get_model_size_mb(baseline_checkpoint)
    int8_size = get_model_size_mb(save_path)
    compression_ratio = fp32_size / max(0.001, int8_size)
    print(f"\n--- PTQ Summary ---")
    print(f"FP32 Baseline Size: {fp32_size:.2f} MB")
    print(f"PTQ INT8 Size:     {int8_size:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}x ({(1 - int8_size/fp32_size)*100:.1f}% reduction)")
    print(f"INT8 Test Accuracy: {int8_acc:.2f}%")

    return quantized_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Post-Training Static Quantization (PTQ)")
    parser.add_argument("--baseline", type=str, default="./checkpoints/baseline_fp32.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/ptq_int8.pt")
    parser.add_argument("--calibration_samples", type=int, default=1000)
    args = parser.parse_args()

    run_ptq(
        baseline_checkpoint=args.baseline,
        save_path=args.save_path,
        num_calibration_samples=args.calibration_samples,
    )
