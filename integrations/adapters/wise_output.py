import json
import re

from core.schemas.clinical import (
    derive_risk_level_from_cbc_evidence,
    extract_request_payload_from_query,
    merge_wise_flags_with_cbc_evidence,
    validate_cbc_payload,
)


def extract_wise_from_text(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {}
    out = {}
    for pattern in (
        r"(?i)risk_level[\"']?\s*:\s*[\"']?(\w+)",
        r"(?i)\*?\s*Risk\s+Level\s*\*?\s*:\s*(\w+)",
        r"(?i)Risk\s+Level:\s*(\w+)",
    ):
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).lower().strip()
            if raw in ("low", "moderate", "high"):
                out["risk_level"] = raw
            elif raw in ("normal", "medium"):
                out["risk_level"] = "moderate"
            else:
                out["risk_level"] = "moderate"
            break
    for pattern in (
        r"(?i)confidence[\"']?\s*:\s*([\d.]+)",
        r"(?i)\*?\s*Confidence\s*\*?\s*:\s*([\d.]+)",
        r"(?i)Confidence:\s*([\d.]+)",
        r"(\d+)\s*%",
    ):
        m = re.search(pattern, text)
        if m:
            try:
                v = float(m.group(1))
                if v > 1:
                    v = v / 100.0
                out["confidence"] = max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                pass
            if "confidence" in out:
                break
    for pattern in (
        r"(?i)flags[\"']?\s*:\s*\[\s*([^\]]*)\s*\]",
        r"(?i)Flags:\s*\[\s*([^\]]*)\s*\]",
    ):
        m = re.search(pattern, text)
        if m:
            inner = m.group(1).strip()
            if not inner:
                out["flags"] = []
            else:
                parts = [p.strip().strip('"\'') for p in re.split(r",", inner)]
                out["flags"] = [p for p in parts if p]
            break
    return out


def normalize_wise_flags(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        seen: set = set()
        out: list = []
        for x in value:
            s = str(x).strip() if x is not None else ""
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out
    if isinstance(value, dict):
        seen = set()
        out = []
        for k, v in value.items():
            key = str(k).strip() if k is not None else ""
            if not key:
                continue
            truthy = False
            if isinstance(v, bool):
                truthy = v
            elif isinstance(v, (int, float)):
                truthy = bool(v) and v != 0
            elif isinstance(v, str):
                truthy = v.strip().lower() in ("true", "1", "yes")
            else:
                truthy = bool(v)
            if truthy and key not in seen:
                seen.add(key)
                out.append(key)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def merge_wise_flag_lists(*lists) -> list:
    seen = set()
    out = []
    for lst in lists:
        if not lst:
            continue
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def ensure_wise_output_schema(output, raw_response: str) -> dict:
    if isinstance(output, list) and output and isinstance(output[0], dict):
        output = output[0]
    if not isinstance(output, dict):
        output = {"response": str(output) if output is not None else raw_response}
    source = (raw_response or "").strip() or (output.get("response") if isinstance(output.get("response"), str) else "")
    extracted = extract_wise_from_text(source) if source else {}
    if output.get("risk_level") not in ("low", "moderate", "high"):
        output["risk_level"] = extracted.get("risk_level") or output.get("risk_level") or "moderate"
        if output["risk_level"] not in ("low", "moderate", "high"):
            output["risk_level"] = "moderate"
    if output.get("confidence") is None:
        output["confidence"] = extracted.get("confidence") if extracted.get("confidence") is not None else 0.5
    else:
        try:
            output["confidence"] = float(output["confidence"])
        except (TypeError, ValueError):
            output["confidence"] = extracted.get("confidence") if extracted.get("confidence") is not None else 0.5
    parsed_flags = normalize_wise_flags(output.get("flags"))
    extracted_flags = normalize_wise_flags(extracted.get("flags"))
    output["flags"] = merge_wise_flag_lists(parsed_flags, extracted_flags)
    return output


def apply_cbc_evidence_to_wise_output(output: dict, input_data: dict) -> dict:
    if not isinstance(output, dict):
        return output
    q = (input_data or {}).get("original_query") or ""
    payload = extract_request_payload_from_query(q)
    if not payload:
        return output
    validated, err = validate_cbc_payload(payload)
    if err or validated is None:
        return output
    llm_flags = normalize_wise_flags(output.get("flags"))
    output["flags"] = merge_wise_flags_with_cbc_evidence(llm_flags, validated)
    output["risk_level"] = derive_risk_level_from_cbc_evidence(output["flags"], validated)
    try:
        conf = float(output.get("confidence", 0.8))
    except (TypeError, ValueError):
        conf = 0.8
    if not output["flags"]:
        output["confidence"] = min(conf, 0.92)
    else:
        output["confidence"] = min(max(conf, 0.72), 0.95)
    return output


def sync_wise_response_footer(output: dict) -> dict:
    if not isinstance(output, dict):
        return output
    response = output.get("response")
    if not isinstance(response, str):
        return output
    response = re.sub(
        r"(?im)^[-*]?\s*Risk\s+Level\s*:\s*.*$",
        f"- Risk Level: {output.get('risk_level', 'low')}",
        response,
    )
    response = re.sub(
        r"(?im)^[-*]?\s*Confidence\s*:\s*.*$",
        f"- Confidence: {output.get('confidence', 0.5)}",
        response,
    )
    response = re.sub(
        r"(?im)^[-*]?\s*Flags\s*:\s*.*$",
        f"- Flags: {json.dumps(output.get('flags', []))}",
        response,
    )
    if "Flags:" not in response:
        response = response.rstrip() + f"\n\n- Flags: {json.dumps(output.get('flags', []))}"
    if "Risk Level:" not in response:
        response = response.rstrip() + f"\n- Risk Level: {output.get('risk_level', 'low')}"
    if "Confidence:" not in response:
        response = response.rstrip() + f"\n- Confidence: {output.get('confidence', 0.5)}"
    output["response"] = response
    return output
