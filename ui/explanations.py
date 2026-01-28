"""
Human-readable explanations for evaluation results.

This module is intentionally UI-oriented and does not modify any evaluation logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _pct(x: float) -> float:
    return x * 100.0


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


@dataclass(frozen=True)
class ComparisonDeltas:
    accuracy_pp: Optional[float]  # percentage points
    latency_pct: Optional[float]  # + means slower, - means faster
    size_pct: Optional[float]     # + means larger, - means smaller


def compute_deltas(
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
) -> ComparisonDeltas:
    base_acc = baseline_metrics.get("Accuracy")
    cand_acc = candidate_metrics.get("Accuracy")
    accuracy_pp = None
    if base_acc is not None and cand_acc is not None:
        accuracy_pp = _pct(cand_acc - base_acc)

    base_lat = baseline_metrics.get("Latency (ms)")
    cand_lat = candidate_metrics.get("Latency (ms)")
    latency_pct = None
    if isinstance(base_lat, (int, float)) and isinstance(cand_lat, (int, float)) and base_lat > 0:
        latency_pct = (cand_lat - base_lat) / base_lat * 100.0

    base_size = baseline_metrics.get("param_MB")
    cand_size = candidate_metrics.get("param_MB")
    size_pct = None
    if isinstance(base_size, (int, float)) and isinstance(cand_size, (int, float)) and base_size > 0:
        size_pct = (cand_size - base_size) / base_size * 100.0

    return ComparisonDeltas(
        accuracy_pp=accuracy_pp,
        latency_pct=latency_pct,
        size_pct=size_pct,
    )


def beginner_conclusion(
    baseline_name: str,
    candidate_name: str,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    *,
    acceptable_accuracy_drop_pp: float = 2.0,
) -> Tuple[str, str]:
    """
    Returns (headline, explanation_paragraph).
    Designed for Beginner Mode: always produces a human-readable explanation.
    """
    d = compute_deltas(baseline_metrics, candidate_metrics)

    # Headline: primary tradeoff
    headline_parts = []
    if d.latency_pct is not None:
        if d.latency_pct < 0:
            headline_parts.append(f"Faster inference ({abs(d.latency_pct):.1f}% faster)")
        elif d.latency_pct > 0:
            headline_parts.append(f"Slower inference ({d.latency_pct:.1f}% slower)")
    if d.accuracy_pp is not None:
        if d.accuracy_pp >= 0:
            headline_parts.append(f"Higher accuracy (+{d.accuracy_pp:.2f}pp)")
        else:
            headline_parts.append(f"Lower accuracy ({d.accuracy_pp:.2f}pp)")

    if not headline_parts:
        headline = f"Comparison: {candidate_name} vs {baseline_name}"
    else:
        headline = " • ".join(headline_parts)

    # Explanation and recommendation
    lines = []
    lines.append(f"Compared **{candidate_name}** against **{baseline_name}**.")

    if d.latency_pct is not None and d.accuracy_pp is not None:
        if d.latency_pct < 0 and d.accuracy_pp < 0:
            lines.append(
                f"This model is faster, but it loses about **{abs(d.accuracy_pp):.2f} percentage points** of accuracy."
            )
        elif d.latency_pct < 0 and d.accuracy_pp >= 0:
            lines.append("This model is faster **and** does not reduce accuracy (nice win-win).")
        elif d.latency_pct >= 0 and d.accuracy_pp >= 0:
            lines.append("This model is slower, but it improves accuracy.")
        else:
            lines.append("This model is slower and less accurate than the baseline.")
    elif d.latency_pct is not None:
        if d.latency_pct < 0:
            lines.append("This model reduces inference time compared to the baseline.")
        else:
            lines.append("This model increases inference time compared to the baseline.")
        if candidate_metrics.get("Accuracy") is None:
            lines.append("Accuracy is not available for this model in the current evaluation pipeline.")
    elif d.accuracy_pp is not None:
        if d.accuracy_pp < 0:
            lines.append("Accuracy is lower than the baseline.")
        else:
            lines.append("Accuracy is higher than the baseline.")

    # Edge readiness heuristic
    edge_reasons = []
    if d.latency_pct is not None and d.latency_pct < 0:
        edge_reasons.append("faster")
    if d.size_pct is not None and d.size_pct < 0:
        edge_reasons.append("smaller")

    if edge_reasons and (d.accuracy_pp is None or d.accuracy_pp >= -acceptable_accuracy_drop_pp):
        lines.append(f"**Conclusion:** This model is a good candidate for **edge deployment** ({', '.join(edge_reasons)}).")
    elif d.accuracy_pp is not None and d.accuracy_pp < -acceptable_accuracy_drop_pp:
        lines.append(
            f"**Conclusion:** Be cautious for edge deployment: the accuracy drop (**{abs(d.accuracy_pp):.2f}pp**) may be noticeable."
        )
    else:
        lines.append("**Conclusion:** Consider your goal (speed vs accuracy) when choosing between these models.")

    return headline, " ".join(lines)

