"""
Recommendation helpers for choosing comparison targets.

This module is UI-facing and only uses existing registries/metadata;
it does not modify evaluation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ui.baseline_registry import get_baseline_registry
from ui.model_inspection import DatasetType, ModelMetadata, TaskType


def normalize_arch_family(architecture: str) -> str:
    a = (architecture or "").lower()
    for key in ["resnet", "mobilenet", "efficientnet", "squeezenet", "distilbert", "bert"]:
        if key in a:
            return key
    return a or "unknown"


@dataclass(frozen=True)
class Recommendation:
    mode: str  # "built_in_baseline" | "uploaded_similar" | "none"
    reason: str
    baseline_model_key: Optional[str] = None
    uploaded_model_name: Optional[str] = None


def recommend_comparison_target(
    metadata: ModelMetadata,
    *,
    saved_models: Dict[str, dict],
) -> Recommendation:
    """
    Decide what to compare an uploaded model against.

    Priority:
    1) Built-in baseline (exact comparison_key match)
    2) Similar uploaded model (same task + same arch family, prefer closest param count if available)
    3) None (analysis-only)
    """
    registry = get_baseline_registry()
    baseline_key = registry.find_baseline(metadata)
    if baseline_key:
        return Recommendation(
            mode="built_in_baseline",
            reason="Found a built-in baseline registered for this architecture/task/dataset.",
            baseline_model_key=baseline_key,
        )

    # Try to find a similar uploaded model
    candidates: List[Tuple[str, dict]] = []
    target_family = normalize_arch_family(metadata.architecture)
    for name, info in (saved_models or {}).items():
        try:
            arch = info.get("architecture", "unknown")
            task = info.get("task", "unknown")
            dataset = info.get("dataset", "unknown")
        except Exception:
            continue

        # Same task is required; dataset match is best-effort
        if task != metadata.task.value:
            continue
        if normalize_arch_family(arch) != target_family:
            continue
        candidates.append((name, info))

    if not candidates:
        return Recommendation(
            mode="none",
            reason="No built-in baseline found, and no similar uploaded models are available.",
        )

    # Prefer dataset match if possible
    same_dataset = [(n, i) for (n, i) in candidates if i.get("dataset") == metadata.dataset.value]
    chosen = same_dataset[0] if same_dataset else candidates[0]

    return Recommendation(
        mode="uploaded_similar",
        reason="No built-in baseline found; using a similar uploaded model (same task and architecture family).",
        uploaded_model_name=chosen[0],
    )

