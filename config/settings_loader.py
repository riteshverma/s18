"""
Centralized Settings Loader

This module provides a single point of access for all runtime configuration.
All backend modules should import settings from here instead of defining their own.

Usage:
    from config.settings_loader import settings, save_settings, reset_settings
    
    # Access settings
    model = settings["models"]["embedding"]
    
    # Update settings
    settings["rag"]["top_k"] = 5
    save_settings()
    
    # Reset to defaults
    reset_settings()
"""

import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

# Paths
CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.json"
DEFAULTS_FILE = CONFIG_DIR / "settings.defaults.json"
PROFILES_DIR = CONFIG_DIR / "profiles"

# --- Settings Cache ---
_settings_cache = None
_persisted_settings_cache = None

_ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_OLLAMA_PORT = 11434
_DEFAULT_LLAMA_CPP_PORT = 8080
_ALLOWED_MCP_MODES = {"legacy", "strict"}
_DEFAULT_MCP_MODE = "legacy"
_DEFAULT_MCP_STARTUP_TIMEOUT_SECONDS = 5
_DEFAULT_MCP_STDIO_CONNECT_TIMEOUT_SECONDS = 120


def _normalize_local_http_base_url(
    base_url: str,
    *,
    setting_name: str,
    default_port: int,
    loopback_only: bool,
) -> str:
    """Validate and normalize local HTTP service base URL."""
    raw = (base_url or "").strip()
    if not raw:
        raise ValueError(f"{setting_name} cannot be empty")

    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{setting_name} must use http or https")
    if not parsed.hostname:
        raise ValueError(f"{setting_name} must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError(f"{setting_name} cannot include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{setting_name} cannot include query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError(f"{setting_name} must not include a path")

    host = parsed.hostname.lower()
    if loopback_only and host not in _ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"{setting_name} host must be loopback (127.0.0.1, localhost, or ::1)"
        )

    port = parsed.port or default_port
    return f"{parsed.scheme}://{host}:{port}"


def _normalize_ollama_base_url(base_url: str, loopback_only: bool) -> str:
    """Validate and normalize Ollama base URL."""
    return _normalize_local_http_base_url(
        base_url,
        setting_name="ollama.base_url",
        default_port=_DEFAULT_OLLAMA_PORT,
        loopback_only=loopback_only,
    )


def _normalize_llama_cpp_base_url(base_url: str, loopback_only: bool) -> str:
    """Validate and normalize llama.cpp base URL."""
    return _normalize_local_http_base_url(
        base_url,
        setting_name="llama_cpp.base_url",
        default_port=_DEFAULT_LLAMA_CPP_PORT,
        loopback_only=loopback_only,
    )


def normalize_llama_cpp_endpoint_path(path: str, endpoint_name: str) -> str:
    """Validate and normalize llama.cpp endpoint path."""
    normalized = (path or "").strip()
    if not normalized:
        raise ValueError(f"llama_cpp.endpoints.{endpoint_name} cannot be empty")
    if not normalized.startswith("/"):
        raise ValueError(f"llama_cpp.endpoints.{endpoint_name} must start with '/'")
    if "?" in normalized or "#" in normalized:
        raise ValueError(
            f"llama_cpp.endpoints.{endpoint_name} cannot include query or fragment"
        )
    return normalized


def validate_ollama_base_url(base_url: str) -> str:
    """Validate and normalize Ollama base URL to loopback-only endpoints."""
    return _normalize_ollama_base_url(base_url, loopback_only=True)


def normalize_runtime_ollama_base_url(base_url: str) -> str:
    """Validate trusted runtime Ollama URL without loopback host restriction."""
    return _normalize_ollama_base_url(base_url, loopback_only=False)


def validate_llama_cpp_base_url(base_url: str) -> str:
    """Validate and normalize llama.cpp base URL to loopback-only endpoints."""
    return _normalize_llama_cpp_base_url(base_url, loopback_only=True)


def normalize_runtime_llama_cpp_base_url(base_url: str) -> str:
    """Validate trusted runtime llama.cpp URL without loopback host restriction."""
    return _normalize_llama_cpp_base_url(base_url, loopback_only=False)


def normalize_mcp_mode(mode: str | None) -> str:
    """Normalize MCP operating mode."""
    raw = (mode or "").strip().lower()
    if not raw:
        return _DEFAULT_MCP_MODE
    if raw not in _ALLOWED_MCP_MODES:
        raise ValueError(
            f"mcp.mode must be one of: {', '.join(sorted(_ALLOWED_MCP_MODES))}"
        )
    return raw


def _deep_merge_dict(base: dict, overlay: dict) -> dict:
    """Recursively merge dict values from overlay into base."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _deepcopy_settings(settings_dict: dict) -> dict:
    return copy.deepcopy(settings_dict)


def _restore_path_from_persisted(target: dict, persisted: dict, path: tuple[str, ...]) -> None:
    """Restore a runtime-only path to its persisted value before writing settings."""
    if not path:
        return
    target_parent = target
    persisted_parent = persisted
    for key in path[:-1]:
        target_parent = target_parent.get(key)
        persisted_parent = persisted_parent.get(key) if isinstance(persisted_parent, dict) else None
        if not isinstance(target_parent, dict):
            return
    leaf = path[-1]
    if isinstance(persisted_parent, dict) and leaf in persisted_parent:
        target_parent[leaf] = _deepcopy_settings(persisted_parent[leaf])
    else:
        target_parent.pop(leaf, None)


def _strip_env_supabase_secrets_for_disk(settings_dict: dict, persisted: dict) -> None:
    """Do not persist Supabase secrets injected from process environment."""
    env_secret_paths = []
    if os.getenv("SUPABASE_ANON_KEY"):
        env_secret_paths.append(("auth", "supabase_anon_key"))
    if os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
        env_secret_paths.append(("supabase_logging", "service_role_key"))
    for path in env_secret_paths:
        _restore_path_from_persisted(settings_dict, persisted, path)


def _strip_hosted_runtime_overrides_for_disk(settings_dict: dict, persisted: dict) -> None:
    """Keep hosted provider rewrites as runtime-only overlays."""
    if not _should_force_gemini_for_hosted(settings_dict):
        return
    for path in (
        ("agent", "model_provider"),
        ("agent", "default_model"),
        ("agent", "overrides"),
        ("models", "insights_provider"),
        ("models", "embedding"),
        ("models", "embedding_provider"),
        ("models", "semantic_chunking"),
        ("models", "image_captioning"),
        ("models", "memory_extraction"),
    ):
        _restore_path_from_persisted(settings_dict, persisted, path)


def settings_for_disk(settings_dict: dict) -> dict:
    """Return a copy of effective settings with runtime-only overlays removed."""
    persisted = _persisted_settings_cache or {}
    disk_settings = _deepcopy_settings(settings_dict)
    _strip_env_supabase_secrets_for_disk(disk_settings, persisted)
    _strip_hosted_runtime_overrides_for_disk(disk_settings, persisted)
    return disk_settings


_REDACTED_SECRET = "[redacted]"


def redact_settings_for_client(settings_dict: dict) -> dict:
    """Redact secret-bearing settings before returning configuration to clients."""
    redacted = _deepcopy_settings(settings_dict)
    for path in (
        ("auth", "supabase_anon_key"),
        ("supabase_logging", "service_role_key"),
    ):
        parent = redacted
        for key in path[:-1]:
            parent = parent.get(key) if isinstance(parent, dict) else None
            if parent is None:
                break
        if isinstance(parent, dict) and parent.get(path[-1]):
            parent[path[-1]] = _REDACTED_SECRET
    return redacted


def _load_profile_settings(profile_name: str) -> dict:
    """Load a JSON profile from config/profiles."""
    candidate = PROFILES_DIR / f"{profile_name}.json"
    if not candidate.exists():
        raise FileNotFoundError(
            f"S18 profile '{profile_name}' not found. Expected: {candidate}"
        )
    return json.loads(candidate.read_text())


def load_settings() -> dict:
    """Load settings from file. Uses cache if already loaded."""
    global _settings_cache, _persisted_settings_cache
    if _settings_cache is None:
        if SETTINGS_FILE.exists():
            persisted_settings = json.loads(SETTINGS_FILE.read_text())
        elif DEFAULTS_FILE.exists():
            # Fall back to defaults if settings.json doesn't exist
            persisted_settings = json.loads(DEFAULTS_FILE.read_text())
            _settings_cache = _deepcopy_settings(persisted_settings)
            save_settings()  # Create settings.json from defaults
        else:
            raise FileNotFoundError(f"No settings files found in {CONFIG_DIR}")
        _persisted_settings_cache = _deepcopy_settings(persisted_settings)
        _settings_cache = _deepcopy_settings(persisted_settings)
        env_profile = os.getenv("S18_PROFILE")
        if env_profile:
            profile_settings = _load_profile_settings(env_profile.strip())
            _settings_cache = _deep_merge_dict(_settings_cache, profile_settings)
        # Allow container/runtime override without editing tracked config files.
        env_ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if env_ollama_base_url:
            _settings_cache.setdefault("ollama", {})
            _settings_cache["ollama"]["base_url"] = env_ollama_base_url
        env_ollama_timeout = os.getenv("OLLAMA_TIMEOUT")
        if env_ollama_timeout and env_ollama_timeout.isdigit():
            _settings_cache.setdefault("ollama", {})
            _settings_cache["ollama"]["timeout"] = int(env_ollama_timeout)
        env_llama_cpp_base_url = os.getenv("LLAMA_CPP_BASE_URL")
        if env_llama_cpp_base_url:
            _settings_cache.setdefault("llama_cpp", {})
            _settings_cache["llama_cpp"]["base_url"] = env_llama_cpp_base_url
        env_llama_cpp_timeout = os.getenv("LLAMA_CPP_TIMEOUT")
        if env_llama_cpp_timeout and env_llama_cpp_timeout.isdigit():
            _settings_cache.setdefault("llama_cpp", {})
            _settings_cache["llama_cpp"]["timeout"] = int(env_llama_cpp_timeout)
        env_run_poll = os.getenv("RUN_POLL_TIMEOUT_SECONDS")
        if env_run_poll and env_run_poll.isdigit():
            _settings_cache["run_poll_timeout_seconds"] = int(env_run_poll)
        env_mcp_tool_timeout = os.getenv("MCP_TOOL_TIMEOUT_SECONDS")
        if env_mcp_tool_timeout and env_mcp_tool_timeout.isdigit():
            _settings_cache.setdefault("mcp", {})
            _settings_cache["mcp"]["tool_timeout_seconds"] = int(env_mcp_tool_timeout)
        env_mcp_mode = os.getenv("MCP_MODE")
        if env_mcp_mode:
            _settings_cache.setdefault("mcp", {})
            _settings_cache["mcp"]["mode"] = normalize_mcp_mode(env_mcp_mode)
        env_mcp_startup_timeout = os.getenv("MCP_STARTUP_TIMEOUT_SECONDS")
        if env_mcp_startup_timeout:
            try:
                startup_timeout = float(env_mcp_startup_timeout)
            except ValueError:
                startup_timeout = None
            if startup_timeout is not None and startup_timeout > 0:
                _settings_cache.setdefault("mcp", {})
                _settings_cache["mcp"]["startup_timeout_seconds"] = startup_timeout
        env_mcp_stdio_connect_timeout = os.getenv("MCP_STDIO_CONNECT_TIMEOUT_SECONDS")
        if env_mcp_stdio_connect_timeout:
            try:
                stdio_timeout = float(env_mcp_stdio_connect_timeout)
            except ValueError:
                stdio_timeout = None
            if stdio_timeout is not None and stdio_timeout > 0:
                _settings_cache.setdefault("mcp", {})
                _settings_cache["mcp"]["stdio_connect_timeout_seconds"] = stdio_timeout
        env_mcp_required_servers = os.getenv("MCP_REQUIRED_SERVERS")
        if env_mcp_required_servers is not None:
            _settings_cache.setdefault("mcp", {})
            _settings_cache["mcp"]["required_servers"] = [
                server.strip()
                for server in env_mcp_required_servers.split(",")
                if server.strip()
            ]
        env_scheduler_tz = os.getenv("SCHEDULER_TIMEZONE")
        if env_scheduler_tz:
            _settings_cache.setdefault("scheduler", {})
            _settings_cache["scheduler"]["timezone"] = env_scheduler_tz
        # Azure OpenAI runtime overrides
        _settings_cache.setdefault("azure_openai", {})
        env_azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if env_azure_endpoint:
            _settings_cache["azure_openai"]["endpoint"] = env_azure_endpoint.rstrip("/")
        env_openai_api_version = os.getenv("OPENAI_API_VERSION")
        if env_openai_api_version:
            _settings_cache["azure_openai"]["api_version"] = env_openai_api_version
        env_azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if env_azure_chat_deployment:
            _settings_cache["azure_openai"]["chat_deployment"] = env_azure_chat_deployment
        env_azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if env_azure_embedding_deployment:
            _settings_cache["azure_openai"]["embedding_deployment"] = env_azure_embedding_deployment
        if os.getenv("AZURE_OPENAI_API_KEY", "").strip():
            _settings_cache["azure_openai"]["api_key_env"] = "AZURE_OPENAI_API_KEY"
        # Supabase/Auth runtime overrides
        env_auth_enabled = os.getenv("AUTH_ENABLED")
        if env_auth_enabled is not None:
            _settings_cache.setdefault("auth", {})
            _settings_cache["auth"]["enabled"] = env_auth_enabled.strip().lower() in {"1", "true", "yes", "on"}
        env_supabase_url = os.getenv("SUPABASE_URL")
        if env_supabase_url:
            _settings_cache.setdefault("auth", {})
            _settings_cache.setdefault("supabase_logging", {})
            _settings_cache["auth"]["supabase_url"] = env_supabase_url
            _settings_cache["supabase_logging"]["supabase_url"] = env_supabase_url
        env_supabase_anon = os.getenv("SUPABASE_ANON_KEY")
        if env_supabase_anon:
            _settings_cache.setdefault("auth", {})
            _settings_cache["auth"]["supabase_anon_key"] = env_supabase_anon
        env_supabase_aud = os.getenv("SUPABASE_JWT_AUDIENCE")
        if env_supabase_aud:
            _settings_cache.setdefault("auth", {})
            _settings_cache["auth"]["supabase_jwt_audience"] = env_supabase_aud
        env_logging_enabled = os.getenv("SUPABASE_LOGGING_ENABLED")
        if env_logging_enabled is not None:
            _settings_cache.setdefault("supabase_logging", {})
            _settings_cache["supabase_logging"]["enabled"] = (
                env_logging_enabled.strip().lower() in {"1", "true", "yes", "on"}
            )
        env_service_role = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if env_service_role:
            _settings_cache.setdefault("supabase_logging", {})
            _settings_cache["supabase_logging"]["service_role_key"] = env_service_role
        env_default_tenant_id = os.getenv("TENANCY_DEFAULT_TENANT_ID")
        if env_default_tenant_id:
            _settings_cache.setdefault("tenancy", {})
            _settings_cache["tenancy"]["default_tenant_id"] = env_default_tenant_id
        env_default_tier = os.getenv("TENANCY_DEFAULT_TIER")
        if env_default_tier:
            _settings_cache.setdefault("tenancy", {})
            _settings_cache["tenancy"]["default_tier"] = env_default_tier.strip().lower()
        env_default_region = os.getenv("TENANCY_DEFAULT_DATA_REGION")
        if env_default_region:
            _settings_cache.setdefault("tenancy", {})
            _settings_cache["tenancy"]["default_data_region"] = env_default_region.strip().lower()
        env_growth_routing = os.getenv("TENANCY_GROWTH_ROUTING_ENABLED")
        if env_growth_routing is not None:
            _settings_cache.setdefault("tenancy", {})
            _settings_cache["tenancy"]["growth_routing_enabled"] = (
                env_growth_routing.strip().lower() in {"1", "true", "yes", "on"}
            )
        # Hosted deploys: prefer Azure OpenAI when configured and provider is not explicitly pinned.
        # Only apply if user has not explicitly pinned a non-Azure provider in settings.json.
        _explicit_provider = _settings_cache.get("agent", {}).get("model_provider", "")
        azure_cfg = _settings_cache.get("azure_openai", {})
        azure_endpoint = (azure_cfg.get("endpoint") or "").strip()
        azure_chat_deployment = (
            os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            or azure_cfg.get("chat_deployment")
            or _settings_cache.get("agent", {}).get("default_model", "")
        )
        azure_key_present = bool(os.getenv("AZURE_OPENAI_API_KEY", "").strip())
        if azure_endpoint and azure_key_present and _explicit_provider not in {"ollama", "gemini", "llama_cpp"}:
            _settings_cache.setdefault("agent", {})
            _settings_cache["agent"]["model_provider"] = "azure_openai"
            if azure_chat_deployment:
                _settings_cache["agent"]["default_model"] = azure_chat_deployment
            _settings_cache.setdefault("models", {})
            _settings_cache["models"]["insights_provider"] = "azure_openai"
            if azure_cfg.get("embedding_deployment"):
                _settings_cache["models"]["embedding"] = azure_cfg["embedding_deployment"]
                _settings_cache["models"]["embedding_provider"] = "azure_openai"
        elif os.getenv("GEMINI_API_KEY", "").strip() and _explicit_provider not in {"ollama", "azure_openai", "llama_cpp"}:
            _settings_cache.setdefault("agent", {})
            _settings_cache["agent"]["model_provider"] = "gemini"
            _apply_gemini_agent_defaults(_settings_cache)
        # Railway / hosted: allow env to override a dev settings.json that pins ollama@localhost.
        env_force_provider = (
            os.getenv("S18_MODEL_PROVIDER") or os.getenv("AGENT_MODEL_PROVIDER") or ""
        ).strip().lower()
        if env_force_provider in {"gemini", "ollama", "azure_openai", "llama_cpp"}:
            _settings_cache.setdefault("agent", {})
            _settings_cache["agent"]["model_provider"] = env_force_provider
            if env_force_provider == "gemini":
                _apply_gemini_agent_defaults(_settings_cache)
        _apply_hosted_gemini_overrides(_settings_cache)
    return _settings_cache


def _apply_gemini_agent_defaults(settings_dict: dict) -> None:
    settings_dict.setdefault("agent", {})
    dm = str(settings_dict["agent"].get("default_model") or "")
    if not dm.lower().startswith("gemini"):
        settings_dict["agent"]["default_model"] = "gemini-2.5-flash"
    settings_dict.setdefault("models", {})
    settings_dict["models"]["insights_provider"] = "gemini"


def _apply_gemini_embedding_defaults(settings_dict: dict) -> None:
    settings_dict.setdefault("models", {})
    settings_dict["models"]["embedding_provider"] = "gemini"
    embedding_model = str(settings_dict["models"].get("embedding", ""))
    if not embedding_model.lower().startswith(("gemini", "text-embedding")):
        settings_dict["models"]["embedding"] = "gemini-embedding-001"


def _is_railway_deploy() -> bool:
    """True when running on Railway (system env injected on every deploy)."""
    return bool(
        os.getenv("RAILWAY_ENVIRONMENT_NAME")
        or os.getenv("RAILWAY_ENVIRONMENT_ID")
        or os.getenv("RAILWAY_SERVICE_ID")
        or os.getenv("RAILWAY_REPLICA_ID")
        or os.getenv("RAILWAY_PUBLIC_DOMAIN")
        or os.getenv("RAILWAY_PROJECT_ID")
    )


def _ollama_points_at_loopback(settings_dict: dict) -> bool:
    base = str((settings_dict.get("ollama") or {}).get("base_url", "")).strip().lower()
    if not base:
        return True
    return any(token in base for token in ("127.0.0.1", "localhost", "[::1]", "::1"))


def _is_local_ollama_profile_active() -> bool:
    """True when operator explicitly selected a local Ollama profile (not Railway)."""
    if _is_railway_deploy():
        return False
    profile = (os.getenv("S18_PROFILE") or "").strip().lower()
    return profile in {
        "local-laptop-gemma",
        "local-laptop-gemma-docker",
        "local-laptop-qwen",
        "privacy-first",
    }


def _should_force_gemini_for_hosted(settings_dict: dict) -> bool:
    if not os.getenv("GEMINI_API_KEY", "").strip():
        return False
    force_flag = os.getenv("S18_FORCE_GEMINI", "").strip().lower()
    if force_flag in {"1", "true", "yes", "on"}:
        return True
    explicit = (os.getenv("S18_MODEL_PROVIDER") or os.getenv("AGENT_MODEL_PROVIDER") or "").strip().lower()
    if explicit == "gemini":
        return True
    if explicit in {"ollama", "azure_openai", "llama_cpp"}:
        return False
    if _is_local_ollama_profile_active():
        return False
    if _is_railway_deploy():
        return True
    # Persisted dev settings on a cloud container (common Railway failure mode).
    return _ollama_points_at_loopback(settings_dict)


def _apply_hosted_gemini_overrides(settings_dict: dict) -> None:
    """
    Cloud/Railway must not call Ollama at localhost:11434 when Gemini is configured.
    Overrides dev settings.json and per-agent ollama overrides (e.g. TestAgent).
    """
    if not _should_force_gemini_for_hosted(settings_dict):
        return
    settings_dict.setdefault("agent", {})
    settings_dict["agent"]["model_provider"] = "gemini"
    _apply_gemini_agent_defaults(settings_dict)
    overrides = settings_dict["agent"].get("overrides")
    if isinstance(overrides, dict):
        for _agent_name, cfg in overrides.items():
            if not isinstance(cfg, dict):
                continue
            if str(cfg.get("model_provider", "")).strip().lower() == "ollama":
                cfg["model_provider"] = "gemini"
                model = str(cfg.get("model", ""))
                if not model.lower().startswith("gemini"):
                    cfg["model"] = "gemini-2.5-flash"
    if _ollama_points_at_loopback(settings_dict):
        settings_dict.setdefault("models", {})
        if str(settings_dict["models"].get("embedding_provider", "")).strip().lower() == "ollama":
            _apply_gemini_embedding_defaults(settings_dict)
        for key in ("semantic_chunking", "image_captioning", "memory_extraction"):
            model_name = str(settings_dict["models"].get(key, ""))
            if model_name and not model_name.lower().startswith("gemini"):
                settings_dict["models"][key] = "gemini-2.5-flash"


# Backward-compatible alias for callers/tests.
_apply_railway_hosted_overrides = _apply_hosted_gemini_overrides

def save_settings() -> None:
    """Save current settings to file."""
    global _settings_cache, _persisted_settings_cache
    if _settings_cache is not None:
        disk_settings = settings_for_disk(_settings_cache)
        SETTINGS_FILE.write_text(json.dumps(disk_settings, indent=2))
        _persisted_settings_cache = _deepcopy_settings(disk_settings)

def reset_settings() -> dict:
    """Reset settings to defaults."""
    global _settings_cache, _persisted_settings_cache
    if DEFAULTS_FILE.exists():
        _settings_cache = json.loads(DEFAULTS_FILE.read_text())
        _persisted_settings_cache = _deepcopy_settings(_settings_cache)
        save_settings()
    return _settings_cache

def reload_settings() -> dict:
    """Force reload settings from disk (useful after external changes)."""
    global _settings_cache, settings
    _settings_cache = None
    loaded = load_settings()
    settings = loaded
    return loaded

# --- Convenience Accessors ---
# These provide direct access to commonly used settings

def get_ollama_url(endpoint: str = "generate") -> str:
    """Get full Ollama URL for a specific endpoint."""
    base = normalize_runtime_ollama_base_url(load_settings()["ollama"]["base_url"])
    if endpoint == "base":
        return base  # Just return base URL without path
    endpoints = {
        "generate": "/api/generate",
        "chat": "/api/chat",
        "embed": "/api/embed",
        "embeddings": "/api/embeddings"
    }
    return f"{base}{endpoints.get(endpoint, '/api/' + endpoint)}"


def get_llama_cpp_url(endpoint: str = "chat_completions") -> str:
    """Get full llama.cpp URL for a specific endpoint."""
    runtime_settings = load_settings()
    cfg = runtime_settings.get("llama_cpp", {})
    base = normalize_runtime_llama_cpp_base_url(
        cfg.get("base_url", "http://127.0.0.1:8080")
    )
    if endpoint == "base":
        return base

    configured_paths = (
        cfg.get("endpoints", {}) if isinstance(cfg.get("endpoints"), dict) else {}
    )
    default_paths = {
        "chat_completions": "/v1/chat/completions",
        "embeddings": "/v1/embeddings",
        "models": "/v1/models",
        "health": "/health",
    }
    endpoint_path = configured_paths.get(
        endpoint, default_paths.get(endpoint, f"/{endpoint}")
    )
    endpoint_path = normalize_llama_cpp_endpoint_path(str(endpoint_path), endpoint)
    return f"{base}{endpoint_path}"

def get_model(purpose: str) -> str:
    """Get model name for a specific purpose."""
    return load_settings()["models"].get(purpose, "gemma4:e4b")


def get_embedding_provider() -> str:
    """Get the configured embedding provider."""
    provider = str(load_settings().get("models", {}).get("embedding_provider", "")).strip().lower()
    return provider or "ollama"


def get_rag_rerank_settings() -> dict:
    """Get normalized RAG reranker settings with safe defaults."""
    rag_settings = load_settings().get("rag", {})
    rerank = rag_settings.get("rerank", {}) if isinstance(rag_settings, dict) else {}
    if not isinstance(rerank, dict):
        rerank = {}

    def _positive_int(value, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _optional_positive_int(value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _positive_float(value, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    provider = str(rerank.get("provider", "local_oss") or "local_oss").strip().lower()
    if provider in {"none", "disabled", "off"}:
        provider = "noop"

    return {
        "enabled": bool(rerank.get("enabled", False)),
        "provider": provider,
        "model": str(rerank.get("model", "") or "").strip(),
        "candidate_k": _positive_int(rerank.get("candidate_k"), 40),
        "top_k": _optional_positive_int(rerank.get("top_k")),
        "timeout_seconds": _positive_float(rerank.get("timeout_seconds"), 8.0),
        "batch_size": _positive_int(rerank.get("batch_size"), 8),
    }


def get_timeout() -> int:
    """Get Ollama timeout in seconds."""
    return load_settings()["ollama"]["timeout"]


def get_llama_cpp_timeout() -> int:
    """Get llama.cpp timeout in seconds."""
    raw = load_settings().get("llama_cpp", {}).get("timeout", 360)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 360
    return value if value > 0 else 360

def get_run_poll_timeout() -> int:
    """Recommended timeout in seconds for clients polling GET /runs/{id}. Full runs often exceed 5 minutes."""
    return load_settings().get("run_poll_timeout_seconds", 900)


def get_mcp_mode() -> str:
    """Get MCP operating mode."""
    mcp_settings = load_settings().get("mcp", {})
    return normalize_mcp_mode(mcp_settings.get("mode"))


def get_mcp_required_servers() -> list[str]:
    """Get MCP servers required for strict readiness."""
    mcp_settings = load_settings().get("mcp", {})
    required = mcp_settings.get("required_servers", [])
    if not isinstance(required, list):
        return []
    return [str(server).strip() for server in required if str(server).strip()]


def get_mcp_startup_timeout() -> float:
    """Get MCP startup timeout in seconds."""
    mcp_settings = load_settings().get("mcp", {})
    timeout = mcp_settings.get(
        "startup_timeout_seconds", _DEFAULT_MCP_STARTUP_TIMEOUT_SECONDS
    )
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError):
        return float(_DEFAULT_MCP_STARTUP_TIMEOUT_SECONDS)
    return timeout_value if timeout_value > 0 else float(_DEFAULT_MCP_STARTUP_TIMEOUT_SECONDS)


def get_mcp_stdio_connect_timeout() -> float:
    """Max seconds to wait for each MCP stdio subprocess to handshake (spawn + initialize)."""
    mcp_settings = load_settings().get("mcp", {})
    timeout = mcp_settings.get(
        "stdio_connect_timeout_seconds", _DEFAULT_MCP_STDIO_CONNECT_TIMEOUT_SECONDS
    )
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError):
        return float(_DEFAULT_MCP_STDIO_CONNECT_TIMEOUT_SECONDS)
    return (
        timeout_value
        if timeout_value > 0
        else float(_DEFAULT_MCP_STDIO_CONNECT_TIMEOUT_SECONDS)
    )

# --- Initialize on import ---
settings = load_settings()
