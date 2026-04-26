import os
import asyncio
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException
import jwt
from jwt import InvalidTokenError
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError

from core.supabase_config import get_supabase_config

# Keep one PyJWKClient per JWKS URL for the lifetime of the process. PyJWKClient
# caches signing keys internally; recreating the client (as an earlier version
# did every 5 minutes) defeated that cache and forced a JWKS network fetch on
# the first request of every window.
_JWKS_CLIENT_CACHE: Dict[str, "PyJWKClient"] = {}


def is_auth_enabled() -> bool:
    return bool(get_supabase_config().get("auth_enabled", False))


def _get_supabase_url() -> str:
    return get_supabase_config().get("url", "")


def _get_expected_issuer(supabase_url: str) -> str:
    return f"{supabase_url}/auth/v1"


def _get_expected_audience() -> Optional[str]:
    return get_supabase_config().get("jwt_audience")


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not isinstance(authorization, str):
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
    jwk_client = _JWKS_CLIENT_CACHE.get(jwks_url)
    if jwk_client is None:
        jwk_client = PyJWKClient(jwks_url)
        _JWKS_CLIENT_CACHE[jwks_url] = jwk_client
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
        claims = await asyncio.to_thread(verify_supabase_jwt, token, supabase_url, audience)
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

