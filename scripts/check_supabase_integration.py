import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import requests


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def check_jwks(supabase_url: str, timeout: float) -> Tuple[bool, str]:
    url = f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, f"JWKS endpoint returned {r.status_code}"
        data = r.json()
        if not isinstance(data, dict) or "keys" not in data:
            return False, "JWKS payload malformed"
        return True, f"JWKS reachable, keys={len(data.get('keys', []))}"
    except Exception as e:
        return False, f"JWKS check error: {e}"


def check_table(supabase_url: str, service_key: str, table: str, timeout: float) -> Tuple[bool, str]:
    url = f"{supabase_url.rstrip('/')}/rest/v1/{table}"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
    }
    params = {"select": "*", "limit": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 200:
            return True, f"Table '{table}' is accessible"
        if r.status_code == 404:
            return False, f"Table '{table}' not found (create the configured Supabase logging tables first)"
        return False, f"Table '{table}' check failed: {r.status_code} {r.text[:200]}"
    except Exception as e:
        return False, f"Table '{table}' check error: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Supabase JWT + table integration for the S18 backend.")
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    load_local_env()

    auth_enabled = os.getenv("AUTH_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    logging_enabled = os.getenv("SUPABASE_LOGGING_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    audience = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated").strip()
    anon_key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    print("=== Supabase Integration Check ===")
    print(f"AUTH_ENABLED={auth_enabled}")
    print(f"SUPABASE_LOGGING_ENABLED={logging_enabled}")
    print(f"SUPABASE_URL configured={bool(supabase_url)}")
    print(f"SUPABASE_JWT_AUDIENCE={audience or '(empty)'}")
    print(f"SUPABASE_ANON_KEY configured={bool(anon_key)}")
    print(f"SUPABASE_SERVICE_ROLE_KEY configured={bool(service_key)}")
    print()

    has_error = False

    if not supabase_url:
        _fail("SUPABASE_URL is missing")
        return 1

    ok, msg = check_jwks(supabase_url, args.timeout)
    if ok:
        _ok(msg)
    else:
        _fail(msg)
        has_error = True

    if auth_enabled and not audience:
        _warn("AUTH_ENABLED=true but SUPABASE_JWT_AUDIENCE is empty")

    if auth_enabled and not anon_key:
        _warn("SUPABASE_ANON_KEY missing (not required for JWKS verification, but usually set for clients)")

    if logging_enabled:
        if not service_key:
            _fail("SUPABASE_LOGGING_ENABLED=true but SUPABASE_SERVICE_ROLE_KEY is missing")
            has_error = True
        else:
            for table in ("agent_request_log", "agent_result_log"):
                ok, msg = check_table(supabase_url, service_key, table, args.timeout)
                if ok:
                    _ok(msg)
                else:
                    _fail(msg)
                    has_error = True
    else:
        _warn("SUPABASE_LOGGING_ENABLED=false; table checks skipped")

    print()
    if has_error:
        _fail("Integration check finished with errors")
        return 2
    _ok("Integration check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
