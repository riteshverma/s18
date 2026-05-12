"""
Keyword-based intent classifier.

Runs before the PlannerAgent — zero model calls, microsecond decision time.
Returns an intent name when enough keywords match, otherwise None (falls through
to the full planner).

For each recognised intent, a hardcoded plan template is provided so the loop
can skip the planner entirely and go straight to DAG execution.

Cloud deployment:  saves one Gemini API call per matched query
Local deployment:  frees one GPU slot per matched query
"""

import re
from config.settings_loader import load_settings
from core.utils import log_step

# ---------------------------------------------------------------------------
# Keyword rules: intent -> list of trigger keywords
# ---------------------------------------------------------------------------
# A query matches an intent when the number of distinct keyword hits >= threshold
# (configured via settings["intent_classifier"]["min_keyword_hits"], default 2)

INTENT_RULES: dict[str, list[str]] = {
    "cbc_interpret": [
        "hemoglobin", "wbc", "platelets", "cbc", "blood count",
        "rbc", "hematocrit", "neutrophil", "lymphocyte", "leukocyte",
    ],
    "lab_lookup": [
        "lab result", "test result", "last labs", "recent labs",
        "lab report", "blood test", "lab values", "lab panel",
    ],
    "mental_health": [
        "phq9", "gad7", "phq-9", "gad-7",
        "depression", "anxiety", "mood", "mental health",
        "suicidal", "self harm", "self-harm", "stress", "wellbeing",
    ],
    "medication": [
        "medication", "prescription", "drug", "dosage", "dose",
        "medicine", "tablet", "capsule", "refill", "pharmacy",
    ],
    "appointment": [
        "book", "schedule", "appointment", "reschedule",
        "cancel appointment", "visit", "consultation", "follow up", "follow-up",
    ],
    "summarize": [
        "summarize", "summary", "tldr", "brief", "overview",
        "key points", "highlight", "condensed",
    ],
}

# ---------------------------------------------------------------------------
# Hardcoded plan templates — one per intent
# These mirror the CBC fast-path pattern already in loop.py
# ---------------------------------------------------------------------------

INTENT_HARDCODED_PLANS: dict[str, dict] = {
    "cbc_interpret": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "EHRDataMinerAgent",
                    "description": "Retrieve patient CBC context from available records.",
                    "reads": ["original_query"],
                    "writes": ["cbc_results"],
                    "status": "pending",
                },
                {
                    "id": "T002",
                    "agent": "ClinicalReasoningAgent",
                    "description": "Interpret CBC results and produce risk, confidence, and flags.",
                    "reads": ["cbc_results"],
                    "writes": ["response"],
                    "status": "pending",
                },
            ],
            "edges": [
                {"source": "Query", "target": "T001"},
                {"source": "T001", "target": "T002"},
            ],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },

    "lab_lookup": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "EHRDataMinerAgent",
                    "description": "Search and retrieve the patient's latest lab results.",
                    "reads": ["original_query"],
                    "writes": ["lab_results"],
                    "status": "pending",
                },
                {
                    "id": "T002",
                    "agent": "SummarizerAgent",
                    "description": "Summarize retrieved lab results for the clinician.",
                    "reads": ["lab_results"],
                    "writes": ["response"],
                    "status": "pending",
                },
            ],
            "edges": [
                {"source": "Query", "target": "T001"},
                {"source": "T001", "target": "T002"},
            ],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },

    "mental_health": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "ThinkerAgent",
                    "description": "Analyze mental-health payload and produce risk, confidence, and flags.",
                    "reads": ["original_query"],
                    "writes": ["response"],
                    "status": "pending",
                }
            ],
            "edges": [{"source": "Query", "target": "T001"}],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },

    "medication": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "EHRDataMinerAgent",
                    "description": "Retrieve patient medication list and prescription history.",
                    "reads": ["original_query"],
                    "writes": ["medication_data"],
                    "status": "pending",
                },
                {
                    "id": "T002",
                    "agent": "ThinkerAgent",
                    "description": "Interpret medication data and answer the clinical question.",
                    "reads": ["medication_data"],
                    "writes": ["response"],
                    "status": "pending",
                },
            ],
            "edges": [
                {"source": "Query", "target": "T001"},
                {"source": "T001", "target": "T002"},
            ],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },

    "appointment": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "ThinkerAgent",
                    "description": "Handle appointment booking, rescheduling, or cancellation request.",
                    "reads": ["original_query"],
                    "writes": ["response"],
                    "status": "pending",
                }
            ],
            "edges": [{"source": "Query", "target": "T001"}],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },

    "summarize": {
        "plan_graph": {
            "nodes": [
                {
                    "id": "T001",
                    "agent": "SummarizerAgent",
                    "description": "Summarize the provided content or query.",
                    "reads": ["original_query"],
                    "writes": ["response"],
                    "status": "pending",
                }
            ],
            "edges": [{"source": "Query", "target": "T001"}],
        },
        "next_step_id": "T001",
        "interpretation_confidence": 1.0,
        "ambiguity_notes": [],
    },
}


def _classifier_settings() -> dict:
    return load_settings().get("intent_classifier", {})


def classify_intent(query: str) -> str | None:
    """
    Return the best-matching intent name, or None if no intent clears the
    keyword threshold.

    Scoring:
    - Multi-word phrases (e.g. "lab result") count as one hit when found as
      a whole phrase.
    - Single keywords require whole-word match (avoids "cbc" matching "cubic").
    - The intent with the highest hit count wins; ties go to None so the
      planner decides.
    """
    cfg = _classifier_settings()
    if not cfg.get("enabled", True):
        return None

    threshold = int(cfg.get("min_keyword_hits", 2))
    q = " ".join((query or "").lower().split())

    scores: dict[str, int] = {}
    for intent, keywords in INTENT_RULES.items():
        count = 0
        for kw in keywords:
            if " " in kw:
                if kw in q:
                    count += 1
            else:
                if re.search(rf"\b{re.escape(kw)}\b", q):
                    count += 1
        if count >= threshold:
            scores[intent] = count

    if not scores:
        return None

    best_intent = max(scores, key=lambda k: scores[k])
    log_step(
        f"Intent classifier: matched '{best_intent}' (score {scores[best_intent]})",
        symbol="🎯",
    )
    return best_intent
