"""Probe the Jev (System One) API with your key to validate the wire format.

Set JEV_API_KEY (or TYPESAFE_API_KEY) in the environment, then:

    python scripts/jev_probe.py "state text to evaluate"

Prints the raw JSON response plus the parsed decisions, so you can
confirm the endpoint, auth and answer shapes before enabling the
DecisionRouter in the app.
"""

from __future__ import annotations

import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.decisions.router import _parse_answers  # noqa: E402
from core.decisions.questions import query_routing_questions  # noqa: E402

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


def main() -> None:
    api_key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("Set JEV_API_KEY (or TYPESAFE_API_KEY) in the environment first.")

    state = " ".join(sys.argv[1:]) or (
        "[patient id: P-102] request: hemoglobin, wbc, platelets — analyze quickly"
    )
    questions = query_routing_questions()

    print(f"POST {ENDPOINT} (model={MODEL}, state={len(state)} chars)")
    resp = httpx.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": MODEL, "state": state, "questions": questions},
        timeout=15.0,
    )
    if resp.status_code >= 400:
        sys.exit(f"HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    print(json.dumps(data, indent=2))

    print("\nParsed decisions:")
    for name, decision in _parse_answers(data.get("answers", {})).items():
        value = decision.value if isinstance(decision.value, str) else round(decision.value, 3)
        print(f"  {name}: {value!r} (confidence={decision.confidence}, p={decision.probabilities})")


if __name__ == "__main__":
    main()
