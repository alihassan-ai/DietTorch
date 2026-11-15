"""
Deliverable 3: Quantization-Aware Training (QAT) Pipeline
Implements:
1. QAT initialization from pretrained FP32 baseline.
2. Layer fusion for QAT (Conv-BN-ReLU).
3. Insertion of Fake-Quantization observation modules (weights and activations).
4. Quantization-aware fine-tuning loop with backpropagation.
5. Conversion to true INT8 quantized model.
6. Serialization to an executable TorchScript checkpoint at checkpoints/qat_int8.pt.
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quantization

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18, get_quantization_backend
from src.dataset import get_cifar10_loaders
from src.utils import evaluate_model, count_parameters, get_model_size_mb


def run_qat(
    baseline_checkpoint="./checkpoints/baseline_fp32.pt",
    save_path="./checkpoints/qat_int8.pt",
    num_steps=300,
    lr=1e-4,
    batch_size=64,
):
    print("=== [Deliverable 3] Quantization-Aware Training (QAT) ===")
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

    # 3. Model Layer Fusion with is_qat=True
    print("Applying QAT layer fusion (Conv2d + BatchNorm2d + ReLU)...")
    model.train()
    model.fuse_model(is_qat=True)

    # 4. Set QConfig and Insert Fake Quantization Modules
    print(f"Configuring default QAT qconfig for '{backend}' backend...")
    model.qconfig = quantization.get_default_qat_qconfig(backend)
    prepared_qat = quantization.prepare_qat(model, inplace=False)
    prepared_qat.train()
    print("Fake quantization modules successfully inserted into weights and activations.")

    # 5. QAT Fine-Tuning Loop
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    optimizer = optim.SGD(prepared_qat.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting QAT fine-tuning for {num_steps} iterations (lr={lr})...")
    step = 0
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_loader:
        if step >= num_steps:
            break
        optimizer.zero_grad()
        outputs = prepared_qat(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        step += 1

        if step % 50 == 0 or step == num_steps:
            print(f"QAT Step [{step}/{num_steps}] - Loss: {running_loss / total:.4f}, Batch Acc: {100.0 * correct / total:.2f}%")

    print("QAT fine-tuning completed successfully.")

    # 6. Convert Trained QAT Model to INT8 Model
    print("Converting trained QAT model to true INT8 quantized model...")
    prepared_qat.eval()
    quantized_qat = quantization.convert(prepared_qat, inplace=False)
    quantized_qat.eval()

    # 7. Evaluate Converted QAT-INT8 on CIFAR-10 Test Set
    print("Evaluating QAT INT8 model on CIFAR-10 test set (CPU)...")
    qat_acc, correct, total = evaluate_model(quantized_qat, test_loader, device="cpu")
    print(f"QAT INT8 Test Accuracy: {qat_acc:.2f}% ({correct}/{total})")

    # 8. Save Executable Model Checkpoint (TorchScript)
    print(f"Exporting executable TorchScript model to {save_path}...")
    sample_input = torch.randn(1, 3, 32, 32)
    scripted_model = torch.jit.trace(quantized_qat, sample_input)
    torch.jit.save(scripted_model, save_path)
    print(f"QAT INT8 checkpoint saved successfully to {save_path}")

    # 9. Verify reload and metrics
    reloaded_model = torch.jit.load(save_path)
    reload_acc, _, _ = evaluate_model(reloaded_model, test_loader, device="cpu")
    print(f"Reloaded Checkpoint Test Accuracy: {reload_acc:.2f}%")

    fp32_size = get_model_size_mb(baseline_checkpoint)
    int8_size = get_model_size_mb(save_path)
    compression_ratio = fp32_size / max(0.001, int8_size)
    print(f"\n--- QAT Summary ---")
    print(f"FP32 Baseline Size: {fp32_size:.2f} MB")
    print(f"QAT INT8 Size:     {int8_size:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}x ({(1 - int8_size/fp32_size)*100:.1f}% reduction)")
    print(f"QAT INT8 Accuracy: {qat_acc:.2f}%")

    return quantized_qat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quantization-Aware Training (QAT)")
    parser.add_argument("--baseline", type=str, default="./checkpoints/baseline_fp32.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/qat_int8.pt")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_qat(
        baseline_checkpoint=args.baseline,
        save_path=args.save_path,
        num_steps=args.steps,
        lr=args.lr,
    )
