import os
import unittest
from unittest.mock import patch

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jwt import ExpiredSignatureError, InvalidAudienceError, InvalidIssuerError
from jwt.exceptions import PyJWKClientError

from core.supabase_auth import require_supabase_user


class SupabaseAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_bearer_token_returns_401(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with self.assertRaises(Exception) as ctx:
                await require_supabase_user(None)
            self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    async def test_invalid_issuer_returns_401(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", side_effect=InvalidIssuerError("bad iss")):
                with self.assertRaises(Exception) as ctx:
                    await require_supabase_user("Bearer token")
                self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    async def test_invalid_audience_returns_401(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", side_effect=InvalidAudienceError("bad aud")):
                with self.assertRaises(Exception) as ctx:
                    await require_supabase_user("Bearer token")
                self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    async def test_expired_token_returns_401(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", side_effect=ExpiredSignatureError("expired")):
                with self.assertRaises(Exception) as ctx:
                    await require_supabase_user("Bearer token")
                self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    async def test_unknown_jwk_kid_returns_401(self):
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", side_effect=PyJWKClientError("kid not found")):
                with self.assertRaises(Exception) as ctx:
                    await require_supabase_user("Bearer token")
                self.assertEqual(getattr(ctx.exception, "status_code", None), 401)

    async def test_valid_token_returns_user_context(self):
        claims = {
            "sub": "user-123",
            "email": "user@example.com",
            "role": "authenticated",
            "iss": "https://demo.supabase.co/auth/v1",
            "aud": "authenticated",
        }
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", return_value=claims):
                out = await require_supabase_user("Bearer token")
                self.assertEqual(out["sub"], "user-123")
                self.assertEqual(out["email"], "user@example.com")


class ProtectedRouteTests(unittest.TestCase):
    def test_protected_route_without_token_returns_401(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(_: dict = Depends(require_supabase_user)):
            return {"ok": True}

        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            client = TestClient(app)
            resp = client.get("/protected")
            self.assertEqual(resp.status_code, 401)

    def test_protected_route_with_valid_token_returns_200(self):
        app = FastAPI()

        @app.get("/protected")
        async def protected(_: dict = Depends(require_supabase_user)):
            return {"ok": True}

        claims = {"sub": "user-123", "iss": "https://demo.supabase.co/auth/v1", "aud": "authenticated"}
        with patch.dict(
            os.environ,
            {"AUTH_ENABLED": "true", "SUPABASE_URL": "https://demo.supabase.co"},
            clear=False,
        ):
            with patch("core.supabase_auth.verify_supabase_jwt", return_value=claims):
                client = TestClient(app)
                resp = client.get("/protected", headers={"Authorization": "Bearer token"})
                self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
