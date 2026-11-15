"""
Deliverable 8: ONNX Model Export Script
Exports:
1. onnx_models/baseline_fp32.onnx
2. onnx_models/unstructured_pruned.onnx
3. onnx_models/structured_pruned.onnx
4. onnx_models/ptq_int8.onnx (using ONNX Runtime dynamic quantization)
"""

import os
import sys
import argparse
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import load_model_for_inference, MODEL_PATHS
from src.utils import get_model_size_mb


def export_models_to_onnx(output_dir="./onnx_models"):
    os.makedirs(output_dir, exist_ok=True)
    print("=== [Deliverable 8] Exporting Models to ONNX Format ===")

    dummy_input = torch.randn(1, 3, 32, 32)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    models_to_export = [
        ("baseline_fp32", "baseline_fp32.onnx"),
        ("unstructured_pruned", "unstructured_pruned.onnx"),
        ("structured_pruned", "structured_pruned.onnx"),
    ]

    for model_key, onnx_filename in models_to_export:
        onnx_path = os.path.join(output_dir, onnx_filename)
        print(f"\nExporting {model_key} to {onnx_path}...")
        model = load_model_for_inference(model_key)
        model.eval()

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Validate with onnx.checker
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        size_mb = get_model_size_mb(onnx_path)
        print(f"  Successfully exported and validated {onnx_filename} ({size_mb:.2f} MB)")

    # Export Quantized INT8 ONNX Model using ONNX Runtime Quantization
    fp32_onnx_path = os.path.join(output_dir, "baseline_fp32.onnx")
    int8_onnx_path = os.path.join(output_dir, "ptq_int8.onnx")
    print(f"\nQuantizing {fp32_onnx_path} to INT8 ONNX -> {int8_onnx_path}...")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = get_model_size_mb(int8_onnx_path)
    print(f"  Successfully created INT8 ONNX model: {int8_onnx_path} ({int8_size_mb:.2f} MB)")

    print("\n--- ONNX Export Summary ---")
    for fname in os.listdir(output_dir):
        if fname.endswith(".onnx"):
            fpath = os.path.join(output_dir, fname)
            print(f"  {fname:<26}: {get_model_size_mb(fpath):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX")
    parser.add_argument("--output_dir", type=str, default="./onnx_models")
    args = parser.parse_args()

    export_models_to_onnx(output_dir=args.output_dir)
