"""
Model architecture definition for CIFAR-10 ResNet-18 supporting:
1. Native FP32 inference and training.
2. Quantization-ready layers (QuantStub, DeQuantStub, FloatFunctional).
3. Layer fusion for Post-Training Static Quantization (PTQ) and Quantization-Aware Training (QAT).
4. Physical architecture reconstruction for Structured Channel Pruning.
"""

import copy
import os
import torch
import torch.nn as nn
import torchvision.models.quantization as qmodels


def get_cifar10_resnet18(pretrained_path=None, num_classes=10):
    """
    Constructs a CIFAR-10 adapted ResNet-18 model with quantizable stubs and FloatFunctional.
    CIFAR-10 images are 32x32, so conv1 is modified to 3x3 stride 1, and maxpool is replaced with Identity.
    """
    model = qmodels.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if pretrained_path is not None and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"[Model] Loaded weights from {pretrained_path}")

    return model


def get_device():
    """Detect and return the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_quantization_backend():
    """
    Detect the supported PyTorch quantization backend.
    Returns 'qnnpack' (preferred on ARM/macOS) or 'fbgemm' (x86).
    """
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        return "qnnpack"
    elif "fbgemm" in supported:
        return "fbgemm"
    return "none"


def apply_structured_channel_pruning(model, prune_ratio=0.3):
    """
    Applies structured channel pruning to intermediate convolutional layers of ResNet-18.
    Specifically prunes `conv1` out_channels and `conv2` in_channels within each BasicBlock
    based on L1-norm filter importance ranking.

    Maintains tensor dimension consistency across residual connections and reconstructs
    the model with physically reduced channel dimensions, eliminating zero-weight padding.
    """
    pruned_model = copy.deepcopy(model)
    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    total_pruned_channels = 0
    total_orig_channels = 0

    for layer_name in layer_names:
        layer = getattr(pruned_model, layer_name)
        for block_idx, block in enumerate(layer):
            conv1 = block.conv1
            bn1 = block.bn1
            conv2 = block.conv2

            orig_out_ch = conv1.out_channels
            num_keep = max(1, int(round(orig_out_ch * (1.0 - prune_ratio))))
            total_orig_channels += orig_out_ch
            total_pruned_channels += (orig_out_ch - num_keep)

            # Compute L1 norm across filter weights: shape (out_channels, in_channels, k, k)
            filter_norms = conv1.weight.data.abs().sum(dim=(1, 2, 3))
            _, keep_indices = torch.topk(filter_norms, num_keep, largest=True, sorted=True)
            keep_indices = keep_indices.sort().values

            # 1. Reconstruct conv1 with reduced out_channels
            new_conv1 = nn.Conv2d(
                conv1.in_channels,
                num_keep,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=(conv1.bias is not None),
            )
            new_conv1.weight.data = conv1.weight.data[keep_indices].clone()
            if conv1.bias is not None:
                new_conv1.bias.data = conv1.bias.data[keep_indices].clone()
            block.conv1 = new_conv1

            # 2. Reconstruct bn1 with reduced num_features
            new_bn1 = nn.BatchNorm2d(
                num_keep,
                eps=bn1.eps,
                momentum=bn1.momentum,
                affine=bn1.affine,
                track_running_stats=bn1.track_running_stats,
            )
            if bn1.affine:
                new_bn1.weight.data = bn1.weight.data[keep_indices].clone()
                new_bn1.bias.data = bn1.bias.data[keep_indices].clone()
            if bn1.track_running_stats:
                new_bn1.running_mean.data = bn1.running_mean.data[keep_indices].clone()
                new_bn1.running_var.data = bn1.running_var.data[keep_indices].clone()
            block.bn1 = new_bn1

            # 3. Reconstruct conv2 with reduced in_channels (dim 1 sliced)
            new_conv2 = nn.Conv2d(
                num_keep,
                conv2.out_channels,
                kernel_size=conv2.kernel_size,
                stride=conv2.stride,
                padding=conv2.padding,
                bias=(conv2.bias is not None),
            )
            new_conv2.weight.data = conv2.weight.data[:, keep_indices].clone()
            if conv2.bias is not None:
                new_conv2.bias.data = conv2.bias.data.clone()
            block.conv2 = new_conv2

    print(
        f"[Structured Pruning] Pruned {total_pruned_channels}/{total_orig_channels} intermediate channels "
        f"({100 * total_pruned_channels / total_orig_channels:.1f}% channel reduction)."
    )
    return pruned_model


MODEL_PATHS = {
    "baseline_fp32": "./checkpoints/baseline_fp32.pt",
    "ptq_int8": "./checkpoints/ptq_int8.pt",
    "qat_int8": "./checkpoints/qat_int8.pt",
    "unstructured_pruned": "./checkpoints/unstructured_pruned.pt",
    "structured_pruned": "./checkpoints/structured_pruned.pt",
}


def load_model_for_inference(model_type, checkpoint_path=None):
    """
    Unified loader for all 5 model variants on CPU.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_PATHS[model_type]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    if model_type in ["ptq_int8", "qat_int8"]:
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
        loaded = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(loaded, torch.nn.Module):
            model = loaded
        else:
            base = get_cifar10_resnet18()
            model = apply_structured_channel_pruning(base, prune_ratio=0.30)
            model.load_state_dict(loaded)
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

