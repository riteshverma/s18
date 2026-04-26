"""
Pydantic schemas for clinical payloads (CBC) in S18 integrations.
Provides validation, physiological bounds, and unit normalization (e.g. g/L -> g/dL).
"""

import json
from typing import Any, Dict, List, Literal, Optional

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

# --- Evidence-based screening thresholds (adults, typical lab units after normalization) ---
# WHO anemia (non-pregnant adults, sea level): Hb < 13 g/dL (men), < 12 g/dL (women).
# When sex is unknown, use the stricter 12 g/dL cutoff to reduce false alarms on borderline values.
WHO_ANEMIA_HEMOGLOBIN_MALE_G_DL = 13.0
WHO_ANEMIA_HEMOGLOBIN_FEMALE_G_DL = 12.0
WHO_ANEMIA_HEMOGLOBIN_UNKNOWN_SEX_G_DL = 12.0

# Critical hemoglobin warranting urgent escalation in many pathways (product safety cap).
CRITICAL_HEMOGLOBIN_G_DL = 7.0

# WBC (canonical 10^3/uL): common adult ref ~4.5–11.0; leukocytosis screening cut ~11.0; leukopenia ~4.0.
WBC_LEUKOCYTOSIS_10_3_PER_UL = 11.0
WBC_LEUKOPENIA_10_3_PER_UL = 4.0

# Platelets (/uL after normalization): thrombocytopenia / thrombocytosis screening cuts (broad).
PLATELETS_THROMBOCYTOPENIA_PER_UL = 150_000.0
PLATELETS_THROMBOCYTOSIS_PER_UL = 450_000.0


class CBCPayload(BaseModel):
    """
    Validated CBC payload for integration workflows. All lab fields are optional to support partial CBCs.
    Hemoglobin can be supplied in g/dL or g/L; unit is normalized to g/dL.
    """

    model_config = ConfigDict(extra="ignore")

    patient_id: Optional[str] = Field(None, description="Patient identifier (often from query prefix)")
    sex: Optional[str] = Field(
        None,
        description="Optional biological sex for WHO anemia thresholds: male|female (or m|f)",
    )
    hemoglobin: Optional[float] = Field(None, description="Hemoglobin in g/dL (or g/L if unit is g/L)")
    wbc: Optional[float] = Field(None, description="WBC in 10^3/uL")
    rbc: Optional[float] = Field(None, description="RBC in 10^6/uL")
    platelets: Optional[float] = Field(None, description="Platelets per uL")
    unit: Optional[str] = Field("g/dL", description="Unit for hemoglobin: g/dL or g/L (normalized to g/dL)")
    normalization_notes: List[str] = Field(default_factory=list, description="Audit trail of unit conversions applied")

    @field_validator("sex", mode="before")
    @classmethod
    def normalize_sex_field(cls, v: Any) -> Any:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return _normalize_sex(v)
        return None

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
        notes: List[str] = list(normalized.get("normalization_notes") or [])

        # Hemoglobin: g/L -> g/dL
        unit = (normalized.get("unit") or "g/dL")
        if isinstance(unit, str) and unit.lower().strip() in ("g/l", "g/liter"):
            hb = normalized.get("hemoglobin")
            if hb is not None and isinstance(hb, (int, float)):
                converted = float(hb) / 10.0
                notes.append(f"hemoglobin: {hb} g/L -> {converted} g/dL")
                normalized["hemoglobin"] = converted
                normalized["unit"] = "g/dL"

        # WBC canonical: 10^3/uL. Accept common /uL values and convert.
        wbc = normalized.get("wbc")
        if isinstance(wbc, (int, float)) and 100.0 < float(wbc) <= 100000.0:
            converted = float(wbc) / 1000.0
            notes.append(f"wbc: {wbc} /uL -> {converted} 10^3/uL")
            normalized["wbc"] = converted

        # RBC canonical: 10^6/uL. Accept /uL absolute values and convert.
        rbc = normalized.get("rbc")
        if isinstance(rbc, (int, float)) and 1000.0 < float(rbc) <= 10000000.0:
            converted = float(rbc) / 1_000_000.0
            notes.append(f"rbc: {rbc} /uL -> {converted} 10^6/uL")
            normalized["rbc"] = converted

        # Platelets canonical: /uL. Accept 10^3/uL style and convert.
        platelets = normalized.get("platelets")
        if isinstance(platelets, (int, float)) and 0.0 < float(platelets) <= 10000.0:
            converted = float(platelets) * 1000.0
            notes.append(f"platelets: {platelets} 10^3/uL -> {converted} /uL")
            normalized["platelets"] = converted

        normalized["normalization_notes"] = notes
        return normalized


class ClinicalAssessment(BaseModel):
    """
    Structured output schema for clinical reasoning agents.
    Validates that agent responses have the expected shape and value ranges.
    """

    model_config = ConfigDict(extra="ignore")

    response: str = Field(..., description="Clinical narrative / interpretation text")
    risk_level: Literal["low", "moderate", "high", "critical"] = Field(
        ..., description="Overall risk classification"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0-1")
    flags: List[str] = Field(default_factory=list, description="Clinical flags (e.g. low_hemoglobin)")
    normalization_notes: List[str] = Field(
        default_factory=list, description="Unit conversion notes from input normalization"
    )


def try_parse_clinical_assessment(raw: Any) -> Optional["ClinicalAssessment"]:
    """
    Best-effort parse of agent output into ClinicalAssessment.
    Returns the model on success, None if the output doesn't match the schema.
    """
    if raw is None:
        return None
    data = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    try:
        return ClinicalAssessment.model_validate(data)
    except Exception:
        return None


def extract_request_payload_from_query(query: str) -> Dict[str, Any]:
    """
    Extract the JSON payload after 'Request:' from a CBC query string.
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


def _normalize_sex(raw: Optional[str]) -> Optional[str]:
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if s in ("male", "m", "man"):
        return "male"
    if s in ("female", "f", "woman"):
        return "female"
    return None


def derive_evidence_based_cbc_flags(cbc: CBCPayload) -> List[str]:
    """
    Derive machine flags from normalized CBC values using conservative adult screening rules.
    Intended to be more stable and patient-appropriate than raw LLM label noise.
    """
    flags: List[str] = []
    sex = _normalize_sex(cbc.sex)

    hb = cbc.hemoglobin
    if hb is not None:
        if sex == "male":
            thr = WHO_ANEMIA_HEMOGLOBIN_MALE_G_DL
        elif sex == "female":
            thr = WHO_ANEMIA_HEMOGLOBIN_FEMALE_G_DL
        else:
            thr = WHO_ANEMIA_HEMOGLOBIN_UNKNOWN_SEX_G_DL
        if float(hb) < thr:
            flags.append("low_hemoglobin")

    wbc = cbc.wbc
    if wbc is not None:
        w = float(wbc)
        if w > WBC_LEUKOCYTOSIS_10_3_PER_UL:
            flags.append("high_wbc")
        elif w < WBC_LEUKOPENIA_10_3_PER_UL:
            flags.append("low_wbc")

    plt = cbc.platelets
    if plt is not None:
        p = float(plt)
        if p < PLATELETS_THROMBOCYTOPENIA_PER_UL:
            flags.append("low_platelets")
        elif p > PLATELETS_THROMBOCYTOSIS_PER_UL:
            flags.append("high_platelets")

    return flags


def filter_llm_flags_against_cbc_evidence(llm_flags: List[str], cbc: CBCPayload) -> List[str]:
    """Drop LLM flag labels that contradict measured CBC values (reduces harmful false alarms)."""
    ev = set(derive_evidence_based_cbc_flags(cbc))
    out: List[str] = []
    for f in llm_flags:
        if f in ("low_hemoglobin",) and "low_hemoglobin" not in ev:
            continue
        if f in ("high_wbc",) and "high_wbc" not in ev:
            continue
        if f in ("low_wbc",) and "low_wbc" not in ev:
            continue
        if f in ("low_platelets",) and "low_platelets" not in ev:
            continue
        if f in ("high_platelets",) and "high_platelets" not in ev:
            continue
        out.append(f)
    return out


def merge_flags_with_cbc_evidence(llm_flags: List[str], cbc: CBCPayload) -> List[str]:
    """
    Union evidence-based flags with LLM flags that are not contradicted by labs.
    Evidence flags are listed first; order is otherwise stable and de-duplicated.
    """
    evidence = derive_evidence_based_cbc_flags(cbc)
    filtered_llm = filter_llm_flags_against_cbc_evidence(llm_flags, cbc)
    seen: set = set()
    merged: List[str] = []
    for item in evidence + filtered_llm:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


def derive_risk_level_from_cbc_evidence(flags: List[str], cbc: CBCPayload) -> str:
    """
    Map evidence flags + key lab severities to a single risk tier for integration consumers.
    Conservative: avoid 'high' unless multiple abnormalities or critical hemoglobin.
    """
    if not flags:
        return "low"
    hb = cbc.hemoglobin
    if hb is not None and float(hb) < CRITICAL_HEMOGLOBIN_G_DL:
        return "high"
    if len(flags) >= 2:
        return "high"
    return "moderate"


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


# Backward-compatible alias used by older integration modules/tests.
def merge_wise_flags_with_cbc_evidence(llm_flags: List[str], cbc: CBCPayload) -> List[str]:
    return merge_flags_with_cbc_evidence(llm_flags, cbc)
