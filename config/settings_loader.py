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

# Paths
CONFIG_DIR = Path(__file__).parent
SETTINGS_FILE = CONFIG_DIR / "settings.json"
DEFAULTS_FILE = CONFIG_DIR / "settings.defaults.json"

# --- Settings Cache ---
_settings_cache = None

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
    base = load_settings()["ollama"]["base_url"]
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

# --- Initialize on import ---
settings = load_settings()
