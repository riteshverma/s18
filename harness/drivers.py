from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from harness.models import HarnessJobRequest, HarnessProvider


@dataclass(frozen=True)
class DriverExecutionPlan:
    provider: HarnessProvider
    binary_name: str
    binary_path: str
    command: List[str]
    stdin_payload: Optional[str]


class HarnessDriverRegistry:
    """Build provider-specific execution plans behind one abstraction."""

    _DEFAULT_PROVIDER_CONFIG: Dict[HarnessProvider, Dict[str, object]] = {
        HarnessProvider.CODEX: {
            "binary": "codex",
            "base_args": [],
            "prompt_mode": "stdin",
            "prompt_flag": None,
        },
        HarnessProvider.CLAUDE: {
            "binary": "claude",
            "base_args": [],
            "prompt_mode": "arg",
            "prompt_flag": "-p",
        },
        HarnessProvider.GEMINI: {
            "binary": "gemini",
            "base_args": [],
            "prompt_mode": "arg",
            "prompt_flag": "-p",
        },
    }

    def __init__(self, settings: dict | None = None) -> None:
        self._settings = settings or {}

    def build_plan(self, request: HarnessJobRequest) -> DriverExecutionPlan:
        provider = request.provider
        provider_config = self._provider_settings(provider)
        binary_name = str(provider_config.get("binary", "")).strip() or provider.value
        binary_path = self._resolve_binary(binary_name)
        if not binary_path:
            raise FileNotFoundError(
                f"{provider.value} CLI not found on PATH. Expected binary: {binary_name}"
            )

        base_args = self._coerce_str_list(provider_config.get("base_args"))
        prompt_mode = str(provider_config.get("prompt_mode", "stdin")).strip().lower()
        prompt_flag = provider_config.get("prompt_flag")
        command = [binary_path, *base_args]
        stdin_payload: Optional[str] = None

        if prompt_mode == "arg":
            if prompt_flag:
                command.extend([str(prompt_flag), request.prompt])
            else:
                command.append(request.prompt)
        else:
            stdin_payload = request.prompt

        command.extend(request.additional_args)
        return DriverExecutionPlan(
            provider=provider,
            binary_name=binary_name,
            binary_path=binary_path,
            command=command,
            stdin_payload=stdin_payload,
        )

    def _provider_settings(self, provider: HarnessProvider) -> Dict[str, object]:
        base = dict(self._DEFAULT_PROVIDER_CONFIG[provider])
        harness_settings = self._settings.get("harness", {})
        providers = harness_settings.get("providers", {}) if isinstance(harness_settings, dict) else {}
        override = providers.get(provider.value, {}) if isinstance(providers, dict) else {}
        if isinstance(override, dict):
            base.update(override)
        return base

    @staticmethod
    def _coerce_str_list(raw: object) -> List[str]:
        if not isinstance(raw, list):
            return []
        return [str(item) for item in raw if str(item).strip()]

    @staticmethod
    def _resolve_binary(binary_name: str) -> Optional[str]:
        explicit = os.getenv(f"S18_{binary_name.upper()}_BIN")
        if explicit and Path(explicit).exists():
            return str(Path(explicit))
        return shutil.which(binary_name)

