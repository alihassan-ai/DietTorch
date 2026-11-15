"""
Deliverable 5: Structured Channel Pruning & Architecture Reconstruction Pipeline
Implements:
1. Channel importance identification via L1-norm filter ranking.
2. Channel pruning with tensor dimension preservation across residual connections.
3. Physical architecture reconstruction (reducing in_channels, out_channels, and BatchNorm stats).
4. Post-pruning fine-tuning pipeline to recover classification accuracy.
5. Serialization of the physically reconstructed model to checkpoints/structured_pruned.pt.
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18, apply_structured_channel_pruning, get_device
from src.dataset import get_cifar10_loaders
from src.utils import evaluate_model, count_parameters, get_model_size_mb


def fine_tune_structured_model(model, train_loader, steps=150, lr=1e-4, device="cpu"):
    """Fine-tunes the physically reconstructed model."""
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    step = 0
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in train_loader:
        if step >= steps:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
        step += 1

        if step % 50 == 0 or step == steps:
            print(f"Fine-Tuning Step [{step}/{steps}] - Loss: {running_loss / total:.4f}, Batch Acc: {100.0 * correct / total:.2f}%")


def run_structured_pruning(
    baseline_checkpoint="./checkpoints/baseline_fp32.pt",
    save_path="./checkpoints/structured_pruned.pt",
    prune_ratio=0.30,
    finetune_steps=150,
    lr=1e-4,
):
    print("=== [Deliverable 5] Structured Channel Pruning & Reconstruction ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # 1. Load trained FP32 baseline
    print(f"Loading baseline FP32 model from {baseline_checkpoint}...")
    baseline_model = get_cifar10_resnet18()
    state_dict = torch.load(baseline_checkpoint, map_location="cpu")
    baseline_model.load_state_dict(state_dict)

    orig_params, _, _ = count_parameters(baseline_model)
    print(f"Original Baseline Parameters: {orig_params:,}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    init_acc, _, _ = evaluate_model(baseline_model, test_loader, device=device)
    print(f"Initial Baseline Accuracy: {init_acc:.2f}%")

    # 2. Channel Pruning & Physical Reconstruction
    print(f"\nPruning {prune_ratio * 100:.1f}% channels from intermediate conv layers and reconstructing architecture...")
    pruned_model = apply_structured_channel_pruning(baseline_model, prune_ratio=prune_ratio)

    pruned_params, _, _ = count_parameters(pruned_model)
    reduction_pct = 100.0 * (1.0 - (pruned_params / orig_params))
    print(f"Reconstructed Parameters: {pruned_params:,} ({reduction_pct:.2f}% physical parameter reduction)")

    # 3. Evaluate Immediately After Pruning (Pre-fine-tuning)
    pre_ft_acc, _, _ = evaluate_model(pruned_model, test_loader, device=device)
    print(f"Pre-Fine-Tuning Accuracy: {pre_ft_acc:.2f}%")

    # 4. Fine-Tuning Pipeline
    print(f"\nStarting fine-tuning for {finetune_steps} steps (lr={lr})...")
    fine_tune_structured_model(pruned_model, train_loader, steps=finetune_steps, lr=lr, device=device)

    # 5. Final Evaluation on CPU
    post_ft_acc, correct, total = evaluate_model(pruned_model, test_loader, device="cpu")
    print(f"\nFinal Structured Pruned Accuracy (CPU): {post_ft_acc:.2f}% ({correct}/{total})")

    # 6. Save Checkpoint
    # For structured pruned model, save full model or state dict (both supported)
    pruned_model.eval()
    pruned_model.to("cpu")
    torch.save(pruned_model, save_path)
    # Also save state dict for flexibility
    state_dict_path = save_path.replace(".pt", "_state_dict.pt")
    torch.save(pruned_model.state_dict(), state_dict_path)
    print(f"Structured pruned model saved to {save_path} and {state_dict_path}")

    # Metrics
    size_mb = get_model_size_mb(save_path)
    print(f"\n--- Structured Pruning Summary ---")
    print(f"Original Parameters:      {orig_params:,}")
    print(f"Reconstructed Parameters: {pruned_params:,} ({reduction_pct:.2f}% reduction)")
    print(f"Model File Size:          {size_mb:.2f} MB")
    print(f"Test Accuracy:            {post_ft_acc:.2f}%")

    return pruned_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Structured Channel Pruning")
    parser.add_argument("--baseline", type=str, default="./checkpoints/baseline_fp32.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/structured_pruned.pt")
    parser.add_argument("--prune_ratio", type=float, default=0.30)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_structured_pruning(
        baseline_checkpoint=args.baseline,
        save_path=args.save_path,
        prune_ratio=args.prune_ratio,
        finetune_steps=args.steps,
        lr=args.lr,
    )
