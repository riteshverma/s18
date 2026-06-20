from typing import Dict


def evaluate_verification_gate(
    *,
    confidence: float,
    risk_level: str = "Moderate",
    remaining_budget_ratio: float = 1.0,
    evidence_count: int = 0,
    ambiguity_count: int = 0,
) -> Dict[str, object]:
    """Return verification tier decision for runtime checks and reflection."""
    risk = (risk_level or "Moderate").strip().lower()
    risk_bonus = 0.0
    if risk == "high":
        risk_bonus = 0.08
    elif risk == "moderate":
        risk_bonus = 0.03

    budget_penalty = 0.0
    if remaining_budget_ratio <= 0.2:
        budget_penalty = -0.08
    elif remaining_budget_ratio <= 0.4:
        budget_penalty = -0.04

    clarify_threshold = max(0.45, min(0.9, 0.7 + risk_bonus))
    needs_clarification = confidence < clarify_threshold and ambiguity_count > 0

    if confidence >= 0.85 and evidence_count >= 2:
        mode = "skip_reflect"
    elif confidence >= 0.65 + budget_penalty:
        mode = "light_check"
    else:
        mode = "full_reflect" if remaining_budget_ratio > 0.2 else "light_check"

    return {
        "mode": mode,
        "needs_clarification": needs_clarification,
        "clarify_threshold": round(clarify_threshold, 3),
        "confidence": float(confidence),
        "remaining_budget_ratio": max(0.0, min(1.0, float(remaining_budget_ratio))),
        "risk_level": risk_level,
    }
