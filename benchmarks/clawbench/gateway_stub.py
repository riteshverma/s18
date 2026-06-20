"""Minimal GatewayClient stand-in for ClawBench scoring without OpenClaw."""

from __future__ import annotations

from typing import Any


class GatewayStub:
    """Stub gateway so ClawBench file/execution verifiers run without OpenClaw."""

    async def __aenter__(self) -> "GatewayStub":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError(f"OpenClaw gateway RPC unavailable for S18 benchmark ({method})")

    async def create_session(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError("OpenClaw gateway unavailable for S18 benchmark judge")

    async def subscribe(self, *args: object, **kwargs: object) -> None:
        return None

    async def send_and_wait(self, *args: object, **kwargs: object):
        raise RuntimeError("OpenClaw gateway unavailable for S18 benchmark judge")

    async def delete_session(self, *args: object, **kwargs: object) -> None:
        return None

    async def get_agent_file(self, *args: object, **kwargs: object) -> str:
        return ""
