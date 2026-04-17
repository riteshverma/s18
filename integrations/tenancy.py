from typing import Any, Dict, Optional

STARTER_TIER = "starter"
GROWTH_TIER = "growth"
ENTERPRISE_HEALTH_TIER = "enterprise-health"


def resolve_tenant_context(
    request_payload: Dict[str, Any],
    user: Optional[Dict[str, Any]],
    tenancy_settings: Dict[str, Any],
) -> Dict[str, str]:
    """Resolve tenant metadata for current request.

    Starter mode defaults to shared infrastructure. We still resolve and attach
    tenant metadata now so we can later route growth tenants to isolated
    schemas/databases without changing public APIs.
    """
    default_tenant_id = str(tenancy_settings.get("default_tenant_id", "default")).strip() or "default"
    default_tier = str(tenancy_settings.get("default_tier", STARTER_TIER)).strip().lower() or STARTER_TIER
    default_region = str(tenancy_settings.get("default_data_region", "in")).strip().lower() or "in"

    tenant_id = str(request_payload.get("tenant_id") or default_tenant_id).strip() or default_tenant_id
    tenant_tier = str(request_payload.get("tenant_tier") or default_tier).strip().lower() or default_tier
    data_region = str(request_payload.get("data_region") or default_region).strip().lower() or default_region

    auth_sub = str((user or {}).get("sub") or "").strip()
    tenant_key = f"{tenant_id}:{tenant_tier}:{data_region}"
    return {
        "tenant_id": tenant_id,
        "tenant_tier": tenant_tier,
        "data_region": data_region,
        "tenant_key": tenant_key,
        "auth_sub": auth_sub,
    }


def can_route_to_growth(tenant_context: Dict[str, str], tenancy_settings: Dict[str, Any]) -> bool:
    """Hook for future per-tenant routing to isolated Growth infrastructure."""
    if not tenancy_settings.get("growth_routing_enabled", False):
        return False
    return tenant_context.get("tenant_tier") in {GROWTH_TIER, ENTERPRISE_HEALTH_TIER}

