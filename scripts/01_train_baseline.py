"""
Deliverable 1: Baseline Model Training and Evaluation Script
Implements and trains / initializes CIFAR-10 ResNet-18 FP32 baseline model.
Saves executable FP32 checkpoint to checkpoints/baseline_fp32.pt.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18, get_device
from src.dataset import get_cifar10_loaders
from src.utils import evaluate_model, count_parameters, get_model_size_mb


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
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

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train / Prepare CIFAR-10 ResNet-18 FP32 Baseline")
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--pretrained_weights", type=str, default="./cifar10_pretrained_resnet18.bin")
    parser.add_argument("--save_path", type=str, default="./checkpoints/baseline_fp32.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = get_device()
    print(f"=== [Deliverable 1] Baseline FP32 ResNet-18 ===")
    print(f"Compute Device: {device}")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Initialize model
    model = get_cifar10_resnet18()
    if os.path.exists(args.pretrained_weights):
        print(f"Loading pretrained weights from {args.pretrained_weights}...")
        sd = torch.load(args.pretrained_weights, map_location="cpu")
        model.load_state_dict(sd)
    else:
        print("Pretrained weights not found. Initializing model for scratch training...")

    model.to(device)

    # Initial evaluation
    initial_acc, _, _ = evaluate_model(model, test_loader, device=device)
    print(f"Pre-training / Loaded Test Accuracy: {initial_acc:.2f}%")

    if args.epochs > 0 and not os.path.exists(args.pretrained_weights):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_acc, _, _ = evaluate_model(model, test_loader, device=device)
            print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {loss:.4f}, Train Acc: {acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # Final evaluation on CPU to match deployment benchmarking standard
    final_acc, correct, total = evaluate_model(model, test_loader, device="cpu")
    print(f"Final FP32 Test Accuracy (CPU): {final_acc:.2f}% ({correct}/{total})")

    # Save model checkpoint
    model.eval()
    model.to("cpu")
    torch.save(model.state_dict(), args.save_path)
    print(f"Checkpoint saved successfully to {args.save_path}")

    # Parameters & Size
    total_p, non_zero_p, sparsity = count_parameters(model)
    size_mb = get_model_size_mb(args.save_path)
    print(f"Total Parameters: {total_p:,}")
    print(f"Active Parameters: {non_zero_p:,}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Disk Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
