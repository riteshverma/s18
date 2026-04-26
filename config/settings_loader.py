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

_ALLOWED_OLLAMA_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_OLLAMA_PORT = 11434
_ALLOWED_MCP_MODES = {"legacy", "strict"}
_DEFAULT_MCP_MODE = "legacy"
_DEFAULT_MCP_STARTUP_TIMEOUT_SECONDS = 5


def _normalize_ollama_base_url(base_url: str, loopback_only: bool) -> str:
    """Validate and normalize Ollama base URL."""
    raw = (base_url or "").strip()
    if not raw:
        raise ValueError("ollama.base_url cannot be empty")

    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("ollama.base_url must use http or https")
    if not parsed.hostname:
        raise ValueError("ollama.base_url must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("ollama.base_url cannot include credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("ollama.base_url cannot include query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("ollama.base_url must not include a path")

    host = parsed.hostname.lower()
    if loopback_only and host not in _ALLOWED_OLLAMA_HOSTS:
        raise ValueError(
            "ollama.base_url host must be loopback (127.0.0.1, localhost, or ::1)"
        )

    port = parsed.port or _DEFAULT_OLLAMA_PORT
    return f"{parsed.scheme}://{host}:{port}"


def validate_ollama_base_url(base_url: str) -> str:
    """Validate and normalize Ollama base URL to loopback-only endpoints."""
    return _normalize_ollama_base_url(base_url, loopback_only=True)


def normalize_runtime_ollama_base_url(base_url: str) -> str:
    """Validate trusted runtime Ollama URL without loopback host restriction."""
    return _normalize_ollama_base_url(base_url, loopback_only=False)


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
    global _settings_cache
    if _settings_cache is None:
        if SETTINGS_FILE.exists():
            _settings_cache = json.loads(SETTINGS_FILE.read_text())
        elif DEFAULTS_FILE.exists():
            # Fall back to defaults if settings.json doesn't exist
            _settings_cache = json.loads(DEFAULTS_FILE.read_text())
            save_settings()  # Create settings.json from defaults
        else:
            raise FileNotFoundError(f"No settings files found in {CONFIG_DIR}")
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
        if azure_endpoint and azure_key_present and _explicit_provider not in {"ollama", "gemini"}:
            _settings_cache.setdefault("agent", {})
            _settings_cache["agent"]["model_provider"] = "azure_openai"
            if azure_chat_deployment:
                _settings_cache["agent"]["default_model"] = azure_chat_deployment
            _settings_cache.setdefault("models", {})
            _settings_cache["models"]["insights_provider"] = "azure_openai"
            if azure_cfg.get("embedding_deployment"):
                _settings_cache["models"]["embedding"] = azure_cfg["embedding_deployment"]
                _settings_cache["models"]["embedding_provider"] = "azure_openai"
        elif os.getenv("GEMINI_API_KEY", "").strip() and _explicit_provider not in {"ollama", "azure_openai"}:
            _settings_cache.setdefault("agent", {})
            _settings_cache["agent"]["model_provider"] = "gemini"
            dm = str(_settings_cache["agent"].get("default_model") or "")
            if not dm.lower().startswith("gemini"):
                _settings_cache["agent"]["default_model"] = "gemini-2.5-flash"
            _settings_cache.setdefault("models", {})
            _settings_cache["models"]["insights_provider"] = "gemini"
    return _settings_cache

def save_settings() -> None:
    """Save current settings to file."""
    global _settings_cache
    if _settings_cache is not None:
        SETTINGS_FILE.write_text(json.dumps(_settings_cache, indent=2))

def reset_settings() -> dict:
    """Reset settings to defaults."""
    global _settings_cache
    if DEFAULTS_FILE.exists():
        _settings_cache = json.loads(DEFAULTS_FILE.read_text())
        save_settings()
    return _settings_cache

def reload_settings() -> dict:
    """Force reload settings from disk (useful after external changes)."""
    global _settings_cache
    _settings_cache = None
    return load_settings()

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

def get_model(purpose: str) -> str:
    """Get model name for a specific purpose."""
    return load_settings()["models"].get(purpose, "gemma3:4b")

def get_timeout() -> int:
    """Get Ollama timeout in seconds."""
    return load_settings()["ollama"]["timeout"]

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

# --- Initialize on import ---
settings = load_settings()
