from dataclasses import dataclass
from typing import Optional, Dict, Any

from ui.model_inspection import inspect_model_metadata
from ui.baseline_registry import get_baseline_registry
from ui.model_loader import load_model, is_model_loadable


@dataclass
class UserIntent:
    mode: str  # "learn" or "upload"
    experience_level: str  # "beginner" or "advanced"
    auto_assumptions: bool = True


@dataclass
class QuantizationPlan:
    model_key: str
    baseline_key: Optional[str]
    comparison_models: list
    explanation_level: str  # "simple" or "technical"
    recommended_actions: list


class ModelIntelligenceEngine:
    """
    This is the brain of your system.
    It decides what to do with models.
    """

    def __init__(self):
        self.registry = get_baseline_registry()

    def analyze_uploaded_model(self, path: str) -> Dict[str, Any]:
        """
        Inspect uploaded model and extract metadata.
        """
        metadata = inspect_model_metadata(path)
        baseline_key = self.registry.find_baseline(metadata)

        return {
            "metadata": metadata,
            "baseline_key": baseline_key,
            "has_baseline": baseline_key is not None
        }

    def decide_quantization_plan(self, model_key: str, intent: UserIntent) -> QuantizationPlan:
        """
        Automatically decide what comparisons and explanations to show.
        """

        # Case 1: Learn mode (built-in models only)
        if intent.mode == "learn":
            comparison_models = self._get_builtin_comparisons(model_key)

            return QuantizationPlan(
                model_key=model_key,
                baseline_key=None,
                comparison_models=comparison_models,
                explanation_level="simple" if intent.experience_level == "beginner" else "technical",
                recommended_actions=["compare_models", "explain_quantization"]
            )

        # Case 2: Upload mode
        analysis = self.analyze_uploaded_model(model_key)
        baseline_key = analysis["baseline_key"]

        comparison_models = []

        if baseline_key:
            comparison_models.append(baseline_key)

        # Auto assumptions (your idea 🔥)
        if intent.auto_assumptions:
            if "quant" in model_key.lower():
                comparison_models += self._get_fp32_versions(baseline_key)
            else:
                comparison_models += self._get_quantized_versions(baseline_key)

        return QuantizationPlan(
            model_key=model_key,
            baseline_key=baseline_key,
            comparison_models=comparison_models,
            explanation_level="simple" if intent.experience_level == "beginner" else "technical",
            recommended_actions=["benchmark", "compare", "suggest_quantization"]
        )

    def _get_builtin_comparisons(self, model_key):
        return [
            k for k in ["distilbert_baseline", "distilbert_dynamic_int8", "distilbert_static_int8"]
            if is_model_loadable(k)
        ]

    def _get_quantized_versions(self, baseline_key):
        if not baseline_key:
            return []
        return [
            k for k in [
                baseline_key.replace("baseline", "dynamic_int8"),
                baseline_key.replace("baseline", "static_int8"),
                baseline_key.replace("baseline", "qat_int8"),
            ]
            if is_model_loadable(k)
        ]

    def _get_fp32_versions(self, baseline_key):
        return [baseline_key] if baseline_key else []