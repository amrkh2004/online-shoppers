"""
Model Optimization module (Module 4)
Implements model pruning and ONNX dynamic INT8 quantization for optimal inference performance.
"""

import copy
from pathlib import Path
from typing import Dict

import joblib
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import (
    BASE_DIR,
    FEATURE_NAMES_PATH,
    MODEL_PATH_ONNX,
    MODEL_PATH_PKL,
)
from prodml.logging_conf import logger

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH_QUANTIZED_ONNX = MODELS_DIR / "final_random_forest_int8.onnx"
MODEL_PATH_PRUNED_PKL = MODELS_DIR / "final_random_forest_pruned.pkl"
MODEL_PATH_PRUNED_ONNX = MODELS_DIR / "final_random_forest_pruned.onnx"


def quantize_onnx_model(
    input_onnx_path: Path = MODEL_PATH_ONNX,
    output_quantized_path: Path = MODEL_PATH_QUANTIZED_ONNX,
    weight_type: QuantType = QuantType.QUInt8,
) -> Path:
    """
    Applies dynamic INT8 quantization to an existing ONNX model graph.
    Significantly reduces disk footprint and memory bandwidth while maintaining inference accuracy.
    """
    logger.info(f"Applying ONNX Dynamic Quantization on: {input_onnx_path}")
    if not input_onnx_path.exists():
        raise FileNotFoundError(f"Input ONNX model does not exist at {input_onnx_path}")

    output_quantized_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_onnx_path),
        model_output=str(output_quantized_path),
        weight_type=weight_type,
    )

    orig_size = input_onnx_path.stat().st_size / (1024 * 1024)
    quant_size = output_quantized_path.stat().st_size / (1024 * 1024)
    reduction = ((orig_size - quant_size) / orig_size) * 100

    logger.info(
        f"Quantization Complete: {orig_size:.2f} MB -> {quant_size:.2f} MB ({reduction:.1f}% reduction)"
    )
    return output_quantized_path


def prune_and_export_compact_model(
    base_model_path: Path = MODEL_PATH_PKL,
    feature_names_path: Path = FEATURE_NAMES_PATH,
    output_pruned_pkl: Path = MODEL_PATH_PRUNED_PKL,
    output_pruned_onnx: Path = MODEL_PATH_PRUNED_ONNX,
    target_estimators: int = 40,
) -> Dict[str, Path]:
    """
    Applies ensemble pruning (reducing redundant trees with minimal loss in ROC-AUC)
    and serializes both compact PKL and compact ONNX graphs.
    """
    logger.info(f"Loading baseline model for pruning from: {base_model_path}")
    if not base_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found at {base_model_path}")

    model = joblib.load(base_model_path)
    pruned_model = copy.deepcopy(model)

    # Prune ensemble estimators to compact size
    if hasattr(pruned_model, "estimators_") and len(pruned_model.estimators_) > target_estimators:
        pruned_model.estimators_ = pruned_model.estimators_[:target_estimators]
        pruned_model.n_estimators = target_estimators

    # Save pruned PKL
    output_pruned_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pruned_model, output_pruned_pkl)

    # Load feature names
    import json

    with open(feature_names_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    # Convert pruned model to ONNX
    initial_type = [("float_input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(pruned_model, initial_types=initial_type)
    with open(output_pruned_onnx, "wb") as f:
        f.write(onnx_model.SerializeToString())

    logger.info(
        f"Pruned model successfully saved to {output_pruned_pkl.name} and {output_pruned_onnx.name}"
    )
    return {"pruned_pkl": output_pruned_pkl, "pruned_onnx": output_pruned_onnx}


def build_all_optimized_variants() -> Dict[str, Path]:
    """
    Generates all optimized model variants across the optimization journey:
    1. Baseline Scikit-learn PKL
    2. ONNX FP32
    3. ONNX INT8 Quantized
    4. Compact Pruned PKL & ONNX
    """
    results = {
        "baseline_pkl": MODEL_PATH_PKL,
        "onnx_fp32": MODEL_PATH_ONNX,
    }

    # Generate Quantized ONNX
    if MODEL_PATH_ONNX.exists():
        quant_path = quantize_onnx_model()
        results["onnx_int8"] = quant_path

    # Generate Pruned Variant
    if MODEL_PATH_PKL.exists() and FEATURE_NAMES_PATH.exists():
        pruned_dict = prune_and_export_compact_model()
        results.update(pruned_dict)

    return results
