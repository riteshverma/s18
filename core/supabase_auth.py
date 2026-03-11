import os
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException
import jwt
from jwt import InvalidTokenError
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError

from config.settings_loader import settings


def _auth_settings() -> Dict[str, Any]:
    return settings.get("auth", {})


def is_auth_enabled() -> bool:
    env_override = os.getenv("AUTH_ENABLED")
    if env_override is not None:
        return env_override.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_auth_settings().get("enabled", False))


def _get_supabase_url() -> str:
    return (
        os.getenv("SUPABASE_URL")
        or _auth_settings().get("supabase_url")
        or ""
    ).rstrip("/")


def _get_expected_issuer(supabase_url: str) -> str:
    return f"{supabase_url}/auth/v1"


def _get_expected_audience() -> Optional[str]:
    return os.getenv("SUPABASE_JWT_AUDIENCE") or _auth_settings().get("supabase_jwt_audience", "authenticated")


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def verify_supabase_jwt(token: str, supabase_url: str, audience: Optional[str]) -> Dict[str, Any]:
    issuer = _get_expected_issuer(supabase_url)
    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
    header = jwt.get_unverified_header(token)
    token_alg = str(header.get("alg", "RS256")).upper()
    allowed_algs = {"RS256", "ES256"}
    if token_alg not in allowed_algs:
        raise InvalidTokenError("Unsupported JWT signing algorithm")
    jwk_client = PyJWKClient(jwks_url)
    signing_key = jwk_client.get_signing_key_from_jwt(token)
    decode_kwargs: Dict[str, Any] = {
        "algorithms": [token_alg],
        "issuer": issuer,
        "options": {"require": ["exp", "iat", "sub", "iss"]},
    }
    if audience:
        decode_kwargs["audience"] = audience
    else:
        decode_kwargs["options"]["verify_aud"] = False
    return jwt.decode(token, signing_key.key, **decode_kwargs)


async def require_supabase_user(
    authorization: Optional[str] = Header(default=None),
    forwarded_authorization: Optional[str] = Header(default=None, alias="X-Forwarded-Authorization"),
) -> Dict[str, Any]:
    """
    Validate Supabase access token using Supabase JWKS and claim checks.
    In local/dev (auth disabled), returns a synthetic user context.
    """
    if not is_auth_enabled():
        return {"sub": "local-dev-user", "email": None, "role": "dev"}

    # Support standard Authorization and a forwarded header for proxy/backend hops.
    token = _extract_bearer_token(authorization) or _extract_bearer_token(forwarded_authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")

    supabase_url = _get_supabase_url()
    if not supabase_url:
        raise HTTPException(status_code=500, detail="Supabase auth not configured on backend")
    audience = _get_expected_audience()

    try:
        claims = verify_supabase_jwt(token, supabase_url, audience)
    except (InvalidTokenError, PyJWKClientError):
        raise HTTPException(status_code=401, detail="Invalid or expired Supabase access token")
    except Exception:
        raise HTTPException(status_code=503, detail="Unable to validate Supabase token")

    return {
        "sub": claims.get("sub"),
        "email": claims.get("email"),
        "role": claims.get("role"),
        "iss": claims.get("iss"),
        "aud": claims.get("aud"),
        "raw": claims,
    }

