"""
Comparison Planner Module

System-driven comparison planning: decides what to compare based on
entry path (Learn vs Upload) and model analysis. UI consumes these
decisions; no evaluation logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ui.baseline_registry import get_baseline_registry
from ui.model_inspection import ModelMetadata
from ui.model_analysis import ModelAnalysisResult


class EntryPath(Enum):
    """User's entry choice."""
    LEARN = "learn"
    UPLOAD = "upload"


class ComparisonIntent(Enum):
    """What comparison the system will perform."""
    LEARN_BUILTIN = "learn_builtin"           # Path A: built-in baseline + variants
    COMPARE_QUANTIZED_TO_BASELINE = "compare_quantized_to_baseline"  # Uploaded quantized → baseline
    COMPARE_BASELINE_TO_VARIANTS = "compare_baseline_to_variants"    # Uploaded FP32 → variants
    ANALYSIS_ONLY = "analysis_only"           # No matching baseline/variants


@dataclass
class ComparisonPlan:
    """
    Result of comparison planning. UI consumes this; evaluation runs elsewhere.
    """
    entry: EntryPath
    intent: ComparisonIntent
    architecture_display: str
    baseline_key: Optional[str] = None
    variant_keys: List[str] = None
    primary_metadata: Optional[ModelMetadata] = None
    primary_model: Any = None  # Only set when upload path already has loaded model
    explanation: str = ""
    can_compare: bool = True
    analysis_only_reason: Optional[str] = None

    def __post_init__(self):
        if self.variant_keys is None:
            self.variant_keys = []


# ---------------------------------------------------------------------------
# Built-in spec: architecture -> { variant_label: model_key }
# Same structure as UI MODEL_REGISTRY; can be passed from app or loaded once.
# ---------------------------------------------------------------------------

def get_builtin_architectures(builtin_spec: Dict[str, Dict[str, str]]) -> List[str]:
    """Return list of architecture names available for Learn path."""
    return list(builtin_spec.keys())


def _baseline_key_for_arch(builtin_spec: Dict[str, Dict[str, str]], architecture: str) -> Optional[str]:
    """Resolve baseline model key for an architecture (prefer 'Baseline' label, else first)."""
    variants = builtin_spec.get(architecture)
    if not variants:
        return None
    if "Baseline" in variants:
        return variants["Baseline"]
    return next(iter(variants.values()))


def _variant_keys_for_arch(builtin_spec: Dict[str, Dict[str, str]], architecture: str) -> List[str]:
    """Return model keys for quantized variants only (exclude baseline). Only includes keys that are loadable (files exist)."""
    from ui.model_loader import is_model_loadable
    variants = builtin_spec.get(architecture)
    if not variants:
        return []
    baseline_key = _baseline_key_for_arch(builtin_spec, architecture)
    return [v for k, v in variants.items() if v != baseline_key and is_model_loadable(v)]


def plan_learn(
    builtin_spec: Dict[str, Dict[str, str]],
    architecture: str,
) -> ComparisonPlan:
    """
    Plan for Path A: Learn about quantization.
    Uses ONLY built-in models; system selects FP32 baseline + all quantized variants.
    Only includes variants whose files exist (is_model_loadable).
    """
    from ui.model_loader import is_model_loadable
    baseline_key = _baseline_key_for_arch(builtin_spec, architecture)
    variant_keys = _variant_keys_for_arch(builtin_spec, architecture)
    if baseline_key and not is_model_loadable(baseline_key):
        baseline_key = None

    if not baseline_key:
        return ComparisonPlan(
            entry=EntryPath.LEARN,
            intent=ComparisonIntent.LEARN_BUILTIN,
            architecture_display=architecture,
            baseline_key=None,
            variant_keys=[],
            explanation=f"No built-in baseline found for {architecture}.",
            can_compare=False,
            analysis_only_reason="No built-in models registered for this architecture.",
        )

    explanation = (
        f"Compare **{architecture}** baseline (FP32) with available quantized variants "
        "to see accuracy, latency, and size tradeoffs."
    )
    return ComparisonPlan(
        entry=EntryPath.LEARN,
        intent=ComparisonIntent.LEARN_BUILTIN,
        architecture_display=architecture,
        baseline_key=baseline_key,
        variant_keys=variant_keys,
        explanation=explanation,
        can_compare=True,
    )


def _quantized_variant_keys_for_metadata(metadata: ModelMetadata) -> List[str]:
    """
    Return built-in model keys that are likely quantized variants for this metadata.
    Does not load models; uses naming heuristics and baseline registry.
    """
    from ui.model_loader import MODEL_REGISTRY

    arch_lower = (metadata.architecture or "").lower()
    if not arch_lower:
        return []

    # Normalize to match registry keys (e.g. ResNet18 -> resnet18)
    arch_token = arch_lower.replace("-", "").replace(" ", "")
    if "resnet" in arch_token:
        arch_token = "resnet18" if "18" in metadata.architecture else "resnet"
    if "distilbert" in arch_token:
        arch_token = "distilbert"

    variant_keys = []
    for model_key in MODEL_REGISTRY.keys():
        if model_key == "user_upload":
            continue
        if arch_token not in model_key.lower():
            continue
        # Skip baseline
        if "baseline" in model_key.lower() and "quant" not in model_key.lower():
            continue
        # Prefer keys that look quantized
        if any(x in model_key.lower() for x in ("ptq", "qat", "dynamic", "int8", "quantized")):
            variant_keys.append(model_key)
        else:
            # Could be another variant (e.g. static); include if not baseline
            if "baseline" not in model_key.lower():
                variant_keys.append(model_key)

    return variant_keys


def plan_upload(
    analysis: ModelAnalysisResult,
) -> ComparisonPlan:
    """
    Plan for Path B: Upload your own model.
    Infers comparison intent from analyzed model:
    - If quantized → compare to FP32 baseline if available
    - If FP32 → compare to known quantized variants
    - If no match → analysis-only with explanation
    """
    metadata = analysis.metadata
    registry = get_baseline_registry()
    baseline_key = registry.find_baseline(metadata)

    # Quantized upload → compare to baseline
    if metadata.is_quantized():
        if baseline_key:
            return ComparisonPlan(
                entry=EntryPath.UPLOAD,
                intent=ComparisonIntent.COMPARE_QUANTIZED_TO_BASELINE,
                architecture_display=metadata.architecture,
                baseline_key=baseline_key,
                variant_keys=[],
                primary_metadata=metadata,
                primary_model=analysis.model,
                explanation=(
                    "Your model is **quantized**. We will compare it to the built-in FP32 baseline "
                    f"for **{metadata.architecture}** to show accuracy, latency, and size tradeoffs."
                ),
                can_compare=True,
            )
        return ComparisonPlan(
            entry=EntryPath.UPLOAD,
            intent=ComparisonIntent.ANALYSIS_ONLY,
            architecture_display=metadata.architecture,
            baseline_key=None,
            variant_keys=[],
            primary_metadata=metadata,
            primary_model=analysis.model,
            explanation=(
                "Your model is **quantized**, but we don't have a built-in FP32 baseline for "
                f"**{metadata.architecture}** ({metadata.task.value} / {metadata.dataset.value}). "
                "You can still view analysis and metrics below."
            ),
            can_compare=False,
            analysis_only_reason="No built-in baseline registered for this architecture/task/dataset.",
        )

    # FP32 upload → compare to quantized variants
    if metadata.is_baseline():
        variant_keys = _quantized_variant_keys_for_metadata(metadata)
        if variant_keys:
            return ComparisonPlan(
                entry=EntryPath.UPLOAD,
                intent=ComparisonIntent.COMPARE_BASELINE_TO_VARIANTS,
                architecture_display=metadata.architecture,
                baseline_key=None,  # primary is the baseline (uploaded)
                variant_keys=variant_keys,
                primary_metadata=metadata,
                primary_model=analysis.model,
                explanation=(
                    "Your model is **FP32 baseline**. We will compare it to built-in quantized "
                    f"variants for **{metadata.architecture}** so you can see the impact of quantization."
                ),
                can_compare=True,
            )
        return ComparisonPlan(
            entry=EntryPath.UPLOAD,
            intent=ComparisonIntent.ANALYSIS_ONLY,
            architecture_display=metadata.architecture,
            baseline_key=None,
            variant_keys=[],
            primary_metadata=metadata,
            primary_model=analysis.model,
            explanation=(
                f"Your model is **FP32**. We don't have built-in quantized variants for "
                f"**{metadata.architecture}** in this app. You can still view analysis and metrics below."
            ),
            can_compare=False,
            analysis_only_reason="No built-in quantized variants for this architecture.",
        )

    # Unknown precision/role
    return ComparisonPlan(
        entry=EntryPath.UPLOAD,
        intent=ComparisonIntent.ANALYSIS_ONLY,
        architecture_display=metadata.architecture,
        baseline_key=baseline_key,
        variant_keys=[],
        primary_metadata=metadata,
        primary_model=analysis.model,
        explanation=(
            "We couldn't determine if your model is baseline or quantized. "
            "Analysis and metrics are shown below; comparison is not available."
        ),
        can_compare=False,
        analysis_only_reason="Precision/quantization role could not be determined.",
    )
