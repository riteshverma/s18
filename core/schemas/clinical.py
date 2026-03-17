"""
Pydantic schemas for clinical payloads (CBC) in the S18/wise-ai integration.
Provides validation, physiological bounds, and unit normalization (e.g. g/L -> g/dL).
"""

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Physiological bounds (after normalization to standard units)
HEMOGLOBIN_G_DL_MIN = 2.0
HEMOGLOBIN_G_DL_MAX = 25.0
WBC_MIN = 0.1  # 10^3/uL
WBC_MAX = 100.0
RBC_MIN = 1.0  # 10^6/uL
RBC_MAX = 10.0
PLATELETS_MIN = 5_000   # /uL
PLATELETS_MAX = 2_000_000


class CBCPayload(BaseModel):
    """
    Validated CBC payload from wise-ai. All lab fields are optional to support partial CBCs.
    Hemoglobin can be supplied in g/dL or g/L; unit is normalized to g/dL.
    """

    model_config = ConfigDict(extra="ignore")

    patient_id: Optional[str] = Field(None, description="Patient identifier (often from query prefix)")
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin in g/dL (or g/L if unit is g/L)")
    wbc: Optional[float] = Field(None, description="WBC in 10^3/uL")
    rbc: Optional[float] = Field(None, description="RBC in 10^6/uL")
    platelets: Optional[float] = Field(None, description="Platelets per uL")
    unit: Optional[str] = Field("g/dL", description="Unit for hemoglobin: g/dL or g/L (normalized to g/dL)")

    @field_validator("hemoglobin")
    @classmethod
    def check_hemoglobin_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < HEMOGLOBIN_G_DL_MIN or v > HEMOGLOBIN_G_DL_MAX:
            raise ValueError(
                f"Hemoglobin value {v} is outside physiological safety bounds "
                f"({HEMOGLOBIN_G_DL_MIN}-{HEMOGLOBIN_G_DL_MAX} g/dL). "
                "If the value is in g/L, set unit='g/L' so it can be normalized."
            )
        return v

    @field_validator("wbc")
    @classmethod
    def check_wbc_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < WBC_MIN or v > WBC_MAX:
            raise ValueError(f"WBC value {v} is outside physiological bounds ({WBC_MIN}-{WBC_MAX} 10^3/uL).")
        return v

    @field_validator("rbc")
    @classmethod
    def check_rbc_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < RBC_MIN or v > RBC_MAX:
            raise ValueError(f"RBC value {v} is outside physiological bounds ({RBC_MIN}-{RBC_MAX} 10^6/uL).")
        return v

    @field_validator("platelets")
    @classmethod
    def check_platelets_range(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < PLATELETS_MIN or v > PLATELETS_MAX:
            raise ValueError(
                f"Platelets value {v} is outside physiological bounds "
                f"({PLATELETS_MIN}-{PLATELETS_MAX} /uL)."
            )
        return v

    @field_validator("unit")
    @classmethod
    def normalize_unit_string(cls, v: Optional[str]) -> str:
        if not v or not v.strip():
            return "g/dL"
        u = v.lower().strip()
        if u in ("g/l", "g/liter"):
            return "g/L"
        return "g/dL"

    @model_validator(mode="before")
    @classmethod
    def normalize_input_units(cls, data: Any) -> Any:
        """
        Normalize supported unit variants before field validation.
        Canonical internal units:
        - hemoglobin: g/dL
        - wbc: 10^3/uL
        - rbc: 10^6/uL
        - platelets: /uL
        """
        if not isinstance(data, dict):
            return data
        normalized = dict(data)

        # Hemoglobin: g/L -> g/dL
        unit = (normalized.get("unit") or "g/dL")
        if isinstance(unit, str) and unit.lower().strip() in ("g/l", "g/liter"):
            hb = normalized.get("hemoglobin")
            if hb is not None and isinstance(hb, (int, float)):
                normalized["hemoglobin"] = float(hb) / 10.0
                normalized["unit"] = "g/dL"

        # WBC canonical: 10^3/uL. Accept common /uL values and convert.
        # Example: 7000 /uL -> 7.0 10^3/uL
        wbc = normalized.get("wbc")
        if isinstance(wbc, (int, float)) and 100.0 < float(wbc) <= 100000.0:
            normalized["wbc"] = float(wbc) / 1000.0

        # RBC canonical: 10^6/uL. Accept /uL absolute values and convert.
        # Example: 4_500_000 /uL -> 4.5 10^6/uL
        rbc = normalized.get("rbc")
        if isinstance(rbc, (int, float)) and 1000.0 < float(rbc) <= 10000000.0:
            normalized["rbc"] = float(rbc) / 1_000_000.0

        # Platelets canonical: /uL. Accept 10^3/uL style and convert.
        # Example: 250 10^3/uL -> 250_000 /uL
        platelets = normalized.get("platelets")
        if isinstance(platelets, (int, float)) and 0.0 < float(platelets) <= 10000.0:
            normalized["platelets"] = float(platelets) * 1000.0

        return normalized


def extract_request_payload_from_query(query: str) -> Dict[str, Any]:
    """
    Extract the JSON payload after 'Request:' from a wise-ai CBC query string.
    E.g. '[Patient ID: x] Request: {"hemoglobin": 7.7, ...}' -> {"hemoglobin": 7.7, ...}
    """
    if not query:
        return {}
    marker = "Request:"
    idx = query.find(marker)
    if idx == -1:
        return {}
    payload_text = query[idx + len(marker) :].strip()
    try:
        payload = json.loads(payload_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def validate_cbc_payload(raw: dict) -> tuple[Optional[CBCPayload], Optional[str]]:
    """
    Validate and normalize a raw CBC payload dict.
    Returns (CBCPayload, None) on success, or (None, error_message) on validation error.
    Empty dict is valid (all fields optional) and yields a model with all None.
    """
    if raw is None or not isinstance(raw, dict):
        return None, "Payload is empty or not a dict"
    try:
        model = CBCPayload.model_validate(raw)
        return model, None
    except Exception as e:
        return None, str(e)
