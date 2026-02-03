"""
Model Analysis Module

Single place for analyzing an uploaded or built-in model. Produces structured
results (metadata, profile, param count, size) used by the comparison planner
and UI. Does not duplicate evaluation logic—only inspection and profiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ui.model_detection import ModelProfile, load_generic_pytorch_model
from ui.model_inspection import ModelMetadata, ModelSource, inspect_model, inspect_model_from_registry
from ui.model_loader import load_model


@dataclass
class ModelAnalysisResult:
    """
    Result of analyzing a model (uploaded or built-in).
    Used by comparison planner and UI; evaluation runs separately.
    """
    metadata: ModelMetadata
    profile: Optional[ModelProfile]
    param_count: int
    size_mb: float
    load_success: bool
    error: Optional[str] = None
    model: Any = None  # PyTorch model if loaded (e.g. for upload path to avoid reload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "param_count": self.param_count,
            "size_mb": self.size_mb,
            "load_success": self.load_success,
            "error": self.error,
        }


def analyze_uploaded_model(
    model_path: str,
    device: str = "cpu",
) -> ModelAnalysisResult:
    """
    Analyze a user-uploaded model file: load, inspect, profile.
    Returns structured result; does not run evaluation.
    """
    model, load_info = load_generic_pytorch_model(model_path, device)

    if not load_info.get("success") or model is None:
        return ModelAnalysisResult(
            metadata=_unknown_metadata(ModelSource.USER, file_path=model_path),
            profile=None,
            param_count=0,
            size_mb=0.0,
            load_success=False,
            error=load_info.get("error", "Failed to load model"),
        )

    metadata = inspect_model(
        model,
        source=ModelSource.USER,
        file_path=model_path,
        device=device,
    )

    try:
        profile = ModelProfile(model, device)
        param_count = profile.param_counts["total"]
        size_mb = profile.size_fp32["size_mb"]
    except Exception as e:
        profile = None
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = (param_count * 4) / (1024 * 1024)  # fp32 bytes

    return ModelAnalysisResult(
        metadata=metadata,
        profile=profile,
        param_count=param_count,
        size_mb=size_mb,
        load_success=True,
        model=model,
    )


def analyze_builtin_model(
    model_key: str,
    device: str = "cpu",
) -> ModelAnalysisResult:
    """
    Analyze a built-in model: load by key, inspect from registry, profile.
    Returns structured result; does not run evaluation.
    """
    try:
        model = load_model(model_key, device)
    except Exception as e:
        return ModelAnalysisResult(
            metadata=_unknown_metadata(ModelSource.INTERNAL, model_key=model_key),
            profile=None,
            param_count=0,
            size_mb=0.0,
            load_success=False,
            error=str(e),
        )

    metadata = inspect_model_from_registry(model_key, model, device)

    try:
        profile = ModelProfile(model, device)
        param_count = profile.param_counts["total"]
        size_mb = profile.size_fp32["size_mb"]
    except Exception:
        profile = None
        param_count = sum(p.numel() for p in model.parameters())
        size_mb = (param_count * 4) / (1024 * 1024)

    return ModelAnalysisResult(
        metadata=metadata,
        profile=profile,
        param_count=param_count,
        size_mb=size_mb,
        load_success=True,
        model=model,
    )


def _unknown_metadata(
    source: ModelSource,
    *,
    model_key: Optional[str] = None,
    file_path: Optional[str] = None,
) -> ModelMetadata:
    """Build minimal metadata when analysis fails."""
    from ui.model_inspection import DatasetType, PrecisionType, QuantizationMethod, TaskType

    return ModelMetadata(
        architecture="Unknown",
        task=TaskType.UNKNOWN,
        dataset=DatasetType.UNKNOWN,
        precision=PrecisionType.UNKNOWN,
        quantization_method=QuantizationMethod.UNKNOWN,
        source=source,
        model_key=model_key,
        file_path=file_path,
        confidence=0.0,
    )
