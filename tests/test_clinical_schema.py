"""Unit tests for core.schemas.clinical (CBC payload validation and normalization)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from pydantic import ValidationError

from core.schemas.clinical import (
    CBCPayload,
    extract_request_payload_from_query,
    validate_cbc_payload,
)


# --- extract_request_payload_from_query ---


def test_extract_request_payload_from_query_valid():
    q = '[Patient ID: test-e2e] Request: {"hemoglobin": 7.7, "wbc": 7000, "rbc": 4.5, "platelets": 250000}'
    out = extract_request_payload_from_query(q)
    assert out == {"hemoglobin": 7.7, "wbc": 7000, "rbc": 4.5, "platelets": 250000}


def test_extract_request_payload_from_query_empty():
    assert extract_request_payload_from_query("") == {}
    assert extract_request_payload_from_query("no Request: here") == {}


def test_extract_request_payload_from_query_with_unit():
    q = '[Patient ID: P123] Request: {"hemoglobin": 145, "unit": "g/L"}'
    out = extract_request_payload_from_query(q)
    assert out == {"hemoglobin": 145, "unit": "g/L"}


# --- CBCPayload validation: valid cases ---


def test_cbc_payload_valid_g_dl():
    m = CBCPayload(hemoglobin=7.7, wbc=7.0, rbc=4.5, platelets=250000)
    assert m.hemoglobin == 7.7
    assert m.unit == "g/dL"


def test_cbc_payload_valid_g_l_normalized():
    """Hemoglobin in g/L is normalized to g/dL (divide by 10)."""
    m = CBCPayload(hemoglobin=145, unit="g/L")
    assert m.hemoglobin == 14.5
    assert m.unit == "g/dL"


def test_cbc_payload_wbc_absolute_normalized():
    """WBC absolute /uL input is normalized to canonical 10^3/uL."""
    m = CBCPayload(wbc=7000)
    assert m.wbc == 7.0


def test_cbc_payload_rbc_absolute_normalized():
    """RBC absolute /uL input is normalized to canonical 10^6/uL."""
    m = CBCPayload(rbc=4_500_000)
    assert m.rbc == 4.5


def test_cbc_payload_platelets_k_normalized():
    """Platelets in 10^3/uL style are normalized to /uL."""
    m = CBCPayload(platelets=250)
    assert m.platelets == 250_000


def test_cbc_payload_partial_empty():
    """Empty or partial payload is valid (all optional)."""
    m = CBCPayload.model_validate({})
    assert m.hemoglobin is None
    assert m.wbc is None


def test_cbc_payload_extra_keys_ignored():
    """extra='ignore' so unknown keys are dropped."""
    m = CBCPayload.model_validate({"hemoglobin": 12.0, "unknown_key": 99})
    assert m.hemoglobin == 12.0
    assert not hasattr(m, "unknown_key") or getattr(m, "unknown_key", None) is None


# --- CBCPayload validation: range errors ---


def test_cbc_payload_hemoglobin_out_of_range_low():
    with pytest.raises(ValidationError) as exc_info:
        CBCPayload(hemoglobin=1.0)
    assert "outside physiological safety bounds" in str(exc_info.value)


def test_cbc_payload_hemoglobin_out_of_range_high():
    with pytest.raises(ValidationError) as exc_info:
        CBCPayload(hemoglobin=30.0)
    assert "outside physiological safety bounds" in str(exc_info.value)


def test_cbc_payload_hemoglobin_g_l_out_of_range_after_normalization():
    """145 g/L -> 14.5 g/dL is valid; 260 g/L -> 26 g/dL would be invalid."""
    with pytest.raises(ValidationError):
        CBCPayload(hemoglobin=260, unit="g/L")


def test_cbc_payload_wbc_out_of_range():
    with pytest.raises(ValidationError):
        CBCPayload(wbc=200_000.0)


def test_cbc_payload_rbc_out_of_range():
    with pytest.raises(ValidationError):
        CBCPayload(rbc=12.0)


def test_cbc_payload_platelets_out_of_range():
    with pytest.raises(ValidationError):
        CBCPayload(platelets=0.1)


# --- validate_cbc_payload helper ---


def test_validate_cbc_payload_success():
    model, err = validate_cbc_payload({"hemoglobin": 12.0, "wbc": 7000})
    assert err is None
    assert model is not None
    assert model.hemoglobin == 12.0
    assert model.wbc == 7.0


def test_validate_cbc_payload_success_g_l():
    model, err = validate_cbc_payload({"hemoglobin": 140, "unit": "g/L"})
    assert err is None
    assert model.hemoglobin == 14.0


def test_validate_cbc_payload_empty_dict_valid():
    """Empty dict is valid (all fields optional) and yields a model with all None."""
    model, err = validate_cbc_payload({})
    assert err is None
    assert model is not None


def test_validate_cbc_payload_failure_not_dict():
    model, err = validate_cbc_payload(None)
    assert model is None
    assert "empty or not a dict" in (err or "").lower()


def test_validate_cbc_payload_failure_invalid_value():
    model, err = validate_cbc_payload({"hemoglobin": 999})
    assert model is None
    assert err is not None
    assert "outside" in (err or "").lower()
