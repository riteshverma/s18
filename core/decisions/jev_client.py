"""HTTP client for TypeSafe AI's Jev (System One) decision API.

Wire format per docs.typesafe.ai/introduction/quickstart:

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer <API_KEY>

    {"model": "jev-latest", "state": "<text>", "questions": {"<id>": {...}}}

Questions are ``{"type": "choice"|"score"|"noul", "instructions": str,
"criteria": ...}``. Answers come back as::

    {"answers": {"<id>": {"choice"|"score"|"noul": <value>,
                          "probabilities": {...}, "confidence": 0..1}},
     "usage": {...}}

Noul answers carry no separate confidence: the 0-1 value itself is the
calibration (near 1 strong yes, near 0 strong no).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import httpx


class JevError(RuntimeError):
    """Raised when the Jev System One API call fails."""


class JevClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://api.typesafe.ai/v1/systemone",
        model: str = "jev-latest",
        timeout_seconds: float = 4.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    async def ask(
        self, state: str, questions: Dict[str, dict]
    ) -> Tuple[Dict[str, dict], dict]:
        """Return (answers_by_id, usage) for one batched System One call."""
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, transport=self._transport
            ) as client:
                resp = await client.post(self.endpoint, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise JevError(f"Jev request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise JevError(
                f"Jev returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise JevError("Jev returned a non-JSON response") from exc
        answers = data.get("answers")
        if not isinstance(answers, dict):
            raise JevError("Jev response is missing an 'answers' object")
        return answers, data.get("usage") or {}
