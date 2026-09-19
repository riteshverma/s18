"""Question definitions for the S18 decision points asked to Jev.

Jev evaluates every question in one request against the same state in
parallel, so a single call per query covers all routing decisions.
Question ids here must match what the router consumers expect
(see ``decision_flag`` / ``decision_choice`` call sites).
"""

from __future__ import annotations

from typing import Dict

QUERY_TYPE_CRITERIA = {
    "cbc": (
        "A structured complete-blood-count lab request or payload "
        "(patient id plus tests like hemoglobin, WBC, platelets)."
    ),
    "mental_health": (
        "A mental-health screening or assessment task "
        "(PHQ-9, GAD-7, depression/anxiety scoring)."
    ),
    "rac": "A risk-adjustment or RAC audit task.",
    "fhir": "A FHIR resource query or interoperability task.",
    "general": "Anything else.",
}


def query_routing_questions() -> Dict[str, dict]:
    """Yes/no + category decisions used by AgentLoop4's planning guards."""
    return {
        "wants_fast_mode": {
            "type": "noul",
            "instructions": (
                "Does the state explicitly or implicitly request fast or quick "
                "execution (e.g. 'fast mode', 'quickly', 'just give me the "
                "result')?"
            ),
        },
        "is_cbc_payload": {
            "type": "noul",
            "instructions": (
                "Does the state contain a structured CBC (complete blood count) "
                "lab request or payload, e.g. a patient id plus requested "
                "values like hemoglobin, WBC and platelets?"
            ),
        },
        "is_mental_health": {
            "type": "noul",
            "instructions": (
                "Is the state a mental-health screening or assessment task "
                "(e.g. PHQ-9, GAD-7, depression or anxiety scoring)?"
            ),
        },
        "query_type": {
            "type": "choice",
            "instructions": "Which category best describes the state?",
            "criteria": QUERY_TYPE_CRITERIA,
        },
    }


def skill_intent_question(registry: Dict[str, dict]) -> Dict[str, dict]:
    """One choice question over the registered skills; 'none' is allowed."""
    criteria = {
        name: (info.get("description") or name)
        for name, info in registry.items()
    }
    criteria["none"] = "No registered skill matches the state."
    return {
        "skill": {
            "type": "choice",
            "instructions": "Which registered skill, if any, best matches the state?",
            "criteria": criteria,
        }
    }
