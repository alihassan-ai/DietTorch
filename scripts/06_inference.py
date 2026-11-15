"""
Deliverable 6: Unified Inference Pipeline
Executes inference across all five model variants:
1. Baseline FP32
2. PTQ INT8
3. QAT INT8
4. Unstructured Pruned FP32
5. Structured Pruned FP32

Supports:
- Running single-sample or batch inference.
- Evaluating classification latency, predictions, and class confidence scores.
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import get_cifar10_resnet18
from src.dataset import get_cifar10_loaders, CIFAR10_CLASSES


MODEL_PATHS = {
    "baseline_fp32": "./checkpoints/baseline_fp32.pt",
    "ptq_int8": "./checkpoints/ptq_int8.pt",
    "qat_int8": "./checkpoints/qat_int8.pt",
    "unstructured_pruned": "./checkpoints/unstructured_pruned.pt",
    "structured_pruned": "./checkpoints/structured_pruned.pt",
}


def load_model_for_inference(model_type, checkpoint_path=None):
    """
    Loads any of the 5 model variants seamlessly for CPU inference.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_PATHS[model_type]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    if model_type in ["ptq_int8", "qat_int8"]:
        # Loaded via TorchScript
        model = torch.jit.load(checkpoint_path, map_location="cpu")
        model.eval()
        return model

    elif model_type in ["baseline_fp32", "unstructured_pruned"]:
        model = get_cifar10_resnet18()
        sd = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        return model

    elif model_type == "structured_pruned":
        # Check if saved as full model object or state dict
        loaded = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(loaded, torch.nn.Module):
            model = loaded
        else:
            # Fallback to reconstructing architecture with 30% pruning
            from src.model import apply_structured_channel_pruning
            base = get_cifar10_resnet18()
            model = apply_structured_channel_pruning(base, prune_ratio=0.30)
            model.load_state_dict(loaded)
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_batch(model, images):
    """
    Runs forward pass on a batch of normalized images and returns class predictions and confidence scores.
    """
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        logits = model(images)
        dt = (time.perf_counter() - t0) * 1000.0  # ms
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)

    return predictions, confidences, dt


def main():
    parser = argparse.ArgumentParser(description="Unified Inference Pipeline for All 5 Models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "baseline_fp32", "ptq_int8", "qat_int8", "unstructured_pruned", "structured_pruned"],
        help="Model variant to test",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="Number of test samples to infer")
    args = parser.parse_args()

    models_to_test = (
        ["baseline_fp32", "ptq_int8", "qat_int8", "unstructured_pruned", "structured_pruned"]
        if args.model == "all"
        else [args.model]
    )

    _, test_loader = get_cifar10_loaders(batch_size=args.num_samples)
    images, targets = next(iter(test_loader))
    images = images[:args.num_samples]
    targets = targets[:args.num_samples]

    print(f"=== [Deliverable 6] Inference Pipeline Execution ===")
    print(f"Testing on {args.num_samples} CIFAR-10 test set images on CPU:\n")

    for m_type in models_to_test:
        print(f"--- Model: {m_type.upper()} ---")
        try:
            model = load_model_for_inference(m_type)
            preds, confs, latency_ms = predict_batch(model, images)
            correct = (preds == targets).sum().item()
            acc = 100.0 * correct / args.num_samples

            print(f"Batch Latency: {latency_ms:.2f} ms ({latency_ms / args.num_samples:.3f} ms/sample)")
            print(f"Batch Accuracy: {acc:.1f}% ({correct}/{args.num_samples} correct)")
            print("Sample Predictions:")
            for i in range(min(5, args.num_samples)):
                pred_label = CIFAR10_CLASSES[preds[i].item()]
                true_label = CIFAR10_CLASSES[targets[i].item()]
                mark = "✓" if preds[i] == targets[i] else "✗"
                print(f"  [{i+1}] True: {true_label:<11} | Pred: {pred_label:<11} (Conf: {confs[i].item()*100:.1f}%) {mark}")
            print()
        except Exception as e:
            print(f"Error executing {m_type}: {e}\n")


if __name__ == "__main__":
    main()
