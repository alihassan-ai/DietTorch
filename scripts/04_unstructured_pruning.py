"""
Deliverable 4: Unstructured Magnitude-Based Pruning Pipeline
Implements:
1. Magnitude-based L1 unstructured pruning on all Conv2d and Linear layers.
2. Iterative pruning across multiple iterations with interleaved fine-tuning to recover accuracy.
3. Weight masking and permanent mask application (prune.remove).
4. Full evaluation on CIFAR-10 test set.
5. Serialization to checkpoints/unstructured_pruned.pt.
"""

import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18, get_device
from src.dataset import get_cifar10_loaders
from src.utils import evaluate_model, count_parameters, get_model_size_mb


def get_prunable_layers(model):
    """Returns list of (module, name) tuples for all Conv2d and Linear layers."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layers.append((module, "weight"))
    return layers


def calculate_global_sparsity(model):
    """Calculates active non-zero parameters and global sparsity percentage."""
    total_elements = 0
    total_zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    return (100.0 * total_zeros / total_elements) if total_elements > 0 else 0.0


def fine_tune_pruned_model(model, train_loader, steps=80, lr=1e-4, device="cpu"):
    """Fine-tunes the pruned model while respecting the zero masks."""
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    step = 0
    for images, targets in train_loader:
        if step >= steps:
            break
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        step += 1


def run_unstructured_pruning(
    baseline_checkpoint="./checkpoints/baseline_fp32.pt",
    save_path="./checkpoints/unstructured_pruned.pt",
    target_sparsity=0.50,
    num_iterations=3,
    finetune_steps_per_iter=80,
    lr=1e-4,
):
    print("=== [Deliverable 4] Iterative Unstructured Pruning ===")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # 1. Load trained FP32 baseline
    print(f"Loading baseline FP32 model from {baseline_checkpoint}...")
    model = get_cifar10_resnet18()
    state_dict = torch.load(baseline_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Initial evaluation
    init_acc, _, _ = evaluate_model(model, test_loader, device=device)
    print(f"Initial Baseline Accuracy: {init_acc:.2f}%")

    # 2. Iterative Pruning Schedule
    # Calculate geometric progression of pruning ratios to reach target_sparsity
    # e.g., for 3 iterations to reach 50%: ~20%, 35%, 50%
    sparsity_steps = [
        target_sparsity * ((i + 1) / num_iterations) for i in range(num_iterations)
    ]

    for it, target_s in enumerate(sparsity_steps, 1):
        print(f"\n--- Pruning Iteration {it}/{num_iterations} (Target Cumulative Sparsity: {target_s * 100:.1f}%) ---")
        
        # Apply L1 unstructured pruning to Conv2d and Linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # If already pruned in previous iteration, remove old hook before re-pruning to target_s
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")
                prune.l1_unstructured(module, name="weight", amount=target_s)

        current_sparsity = calculate_global_sparsity(model)
        pre_ft_acc, _, _ = evaluate_model(model, test_loader, device=device)
        print(f"Post-Pruning Sparsity: {current_sparsity:.2f}%, Test Acc: {pre_ft_acc:.2f}%")

        # Fine-tune to recover accuracy
        print(f"Fine-tuning for {finetune_steps_per_iter} steps (lr={lr})...")
        fine_tune_pruned_model(model, train_loader, steps=finetune_steps_per_iter, lr=lr, device=device)
        post_ft_acc, _, _ = evaluate_model(model, test_loader, device=device)
        print(f"Post-Fine-Tuning Test Acc: {post_ft_acc:.2f}%")

    # 3. Make Pruning Permanent
    print("\nMaking unstructured pruning permanent (removing forward pre-hooks)...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")

    # 4. Final Evaluation & Verification
    final_sparsity = calculate_global_sparsity(model)
    final_acc, correct, total = evaluate_model(model, test_loader, device="cpu")
    print(f"Final Permanent Sparsity: {final_sparsity:.2f}%")
    print(f"Final Pruned Test Accuracy (CPU): {final_acc:.2f}% ({correct}/{total})")

    # 5. Save Checkpoint
    model.eval()
    model.to("cpu")
    torch.save(model.state_dict(), save_path)
    print(f"Unstructured pruned checkpoint saved to {save_path}")

    # Metrics
    total_p, non_zero_p, sparsity_pct = count_parameters(model)
    size_mb = get_model_size_mb(save_path)
    print(f"\n--- Unstructured Pruning Summary ---")
    print(f"Total Parameters:      {total_p:,}")
    print(f"Active Non-Zero Params: {non_zero_p:,}")
    print(f"Weight Sparsity:       {sparsity_pct:.2f}%")
    print(f"Model File Size:       {size_mb:.2f} MB")
    print(f"Test Accuracy:         {final_acc:.2f}%")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Iterative Unstructured Pruning")
    parser.add_argument("--baseline", type=str, default="./checkpoints/baseline_fp32.pt")
    parser.add_argument("--save_path", type=str, default="./checkpoints/unstructured_pruned.pt")
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    run_unstructured_pruning(
        baseline_checkpoint=args.baseline,
        save_path=args.save_path,
        target_sparsity=args.sparsity,
        num_iterations=args.iterations,
        finetune_steps_per_iter=args.steps,
    )
