# Settings Router - Manages system configuration and dependencies (Ollama)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import os

# Import from config system
from config.settings_loader import (
    get_llama_cpp_timeout,
    get_llama_cpp_url,
    reload_settings,
    save_settings,
    reset_settings,
    get_ollama_url,
    normalize_llama_cpp_endpoint_path,
    normalize_runtime_llama_cpp_base_url,
    validate_ollama_base_url,
    validate_llama_cpp_base_url,
    load_settings,
    _is_railway_deploy,
    _should_force_gemini_for_hosted,
)
from shared.state import settings

router = APIRouter()


# === SETTINGS API ENDPOINTS ===

@router.get("/settings/runtime")
async def get_runtime_settings():
    """Effective settings after env/Railway/profile overrides (use for hosted debugging)."""
    loaded = load_settings()
    agent = loaded.get("agent", {})
    return {
        "status": "success",
        "model_provider": agent.get("model_provider"),
        "default_model": agent.get("default_model"),
        "ollama_base_url": loaded.get("ollama", {}).get("base_url"),
        "insights_provider": loaded.get("models", {}).get("insights_provider"),
        "railway_detected": _is_railway_deploy(),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "hosted_gemini_forced": _should_force_gemini_for_hosted(loaded),
    }


@router.get("/settings")
async def get_settings():
    """Get all current settings from config/settings.json"""
    try:
        # Force reload to get latest from disk
        current_settings = reload_settings()
        return {"status": "success", "settings": current_settings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load settings: {str(e)}")


class UpdateSettingsRequest(BaseModel):
    settings: dict


@router.put("/settings")
async def update_settings(request: UpdateSettingsRequest):
    """Update settings and save to config/settings.json
    
    Note: Some settings require re-indexing (chunk_size, chunk_overlap, etc.)
    or server restart to take effect.
    """
    try:
        ollama_settings = request.settings.get("ollama", {})
        if isinstance(ollama_settings, dict) and "base_url" in ollama_settings:
            try:
                ollama_settings["base_url"] = validate_ollama_base_url(
                    str(ollama_settings["base_url"])
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        llama_cpp_settings = request.settings.get("llama_cpp", {})
        if isinstance(llama_cpp_settings, dict):
            if "base_url" in llama_cpp_settings:
                try:
                    llama_cpp_settings["base_url"] = validate_llama_cpp_base_url(
                        str(llama_cpp_settings["base_url"])
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
            endpoint_settings = llama_cpp_settings.get("endpoints", {})
            if isinstance(endpoint_settings, dict):
                for endpoint_name, endpoint_path in endpoint_settings.items():
                    try:
                        endpoint_settings[endpoint_name] = normalize_llama_cpp_endpoint_path(
                            str(endpoint_path), str(endpoint_name)
                        )
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc))

        # Use shared global settings
        global settings 
        
        # Deep merge incoming settings with existing
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict) and value:
                    deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        # Reload potentially stale global settings just in case
        settings = reload_settings()
        deep_merge(settings, request.settings)
        save_settings()
        
        # Identify settings that require action
        warnings = []
        rag_keys = ["chunk_size", "chunk_overlap", "max_chunk_length", "semantic_word_limit"]
        if "rag" in request.settings:
            for key in rag_keys:
                if key in request.settings["rag"]:
                    warnings.append(f"Changed '{key}' - requires re-indexing documents to take effect")
        
        if "models" in request.settings:
            # Agent models take effect on next run, but RAG models might need more
            warnings.append("Agent model changes will take effect on the next run.")
            warnings.append("RAG model changes take effect on next document processing or server restart")
        
        return {
            "status": "success",
            "message": "Settings saved successfully",
            "warnings": warnings if warnings else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {str(e)}")


@router.post("/settings/reset")
async def reset_to_defaults():
    """Reset all settings to default values from config/settings.defaults.json"""
    try:
        reset_settings()
        return {"status": "success", "message": "Settings reset to defaults"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {str(e)}")


@router.post("/settings/restart")
async def restart_server():
    """Return instructions for manual restart.
    
    Note: Automatic restart doesn't work reliably with npm run dev:all / concurrently.
    The proper way is to manually Ctrl+C and restart.
    """
    return {
        "status": "manual_required",
        "message": "Automatic restart is not supported. Please manually restart the server.",
        "instructions": [
            "1. Press Ctrl+C in the terminal running npm run dev:all",
            "2. Run: npm run dev:all",
            "3. Refresh the browser"
        ]
    }


# === OLLAMA API ENDPOINTS ===

@router.get("/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models from local instance"""
    try:
        ollama_url = get_ollama_url("base")
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to connect to Ollama")
        
        data = response.json()
        models = []
        for model in data.get("models", []):
            name = model.get("name", "")
            size_bytes = model.get("size", 0)
            size_gb = round(size_bytes / (1024**3), 2) if size_bytes else 0
            
            # Get family info from Ollama response
            details = model.get("details", {})
            families = details.get("families", [])
            
            # Infer capabilities from model name AND family
            capabilities = set()
            name_lower = name.lower()
            
            # Embedding models
            if "embed" in name_lower or "nomic" in name_lower or "nomic-bert" in families:
                capabilities.add("embedding")
            
            # Vision/multimodal models - check for explicit vision families or name patterns
            vision_families = ["clip", "qwen3vl", "llava"]
            vision_names = ["vl", "vision", "llava", "moondream", "gemma3", "gemma4"]
            
            if any(f in families for f in vision_families) or any(v in name_lower for v in vision_names):
                capabilities.add("text")
                capabilities.add("image")
            else:
                capabilities.add("text")
            
            models.append({
                "name": name,
                "size_gb": size_gb,
                "capabilities": list(capabilities),
                "modified_at": model.get("modified_at", "")
            })
        
        return {"status": "success", "models": models}
        
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Ollama is not running. Please start Ollama.")

class PullModelRequest(BaseModel):
    name: str

@router.post("/ollama/pull")
async def pull_ollama_model(request: PullModelRequest):
    """Pull a new model from Ollama registry (starts async download)"""
    try:
        ollama_url = get_ollama_url("base")
        # Use streaming=False for now, just initiate the pull
        response = requests.post(
            f"{ollama_url}/api/pull",
            json={"name": request.name, "stream": False},
            timeout=600  # 10 min timeout for large models
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Failed to pull model: {response.text}")
        
        return {"status": "success", "message": f"Model '{request.name}' pulled successfully"}
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Model pull timed out - try from terminal")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llama_cpp/models")
async def get_llama_cpp_models():
    """Get list of models from llama.cpp OpenAI-compatible endpoint."""
    try:
        response = requests.get(get_llama_cpp_url("models"), timeout=10)
        response.raise_for_status()
        data = response.json()
        models = []
        for item in data.get("data", []):
            name = item.get("id", "")
            if not name:
                continue
            models.append(
                {
                    "name": name,
                    "capabilities": ["text", "embedding"],
                    "owned_by": item.get("owned_by", "llama.cpp"),
                }
            )
        return {"status": "success", "models": models}
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="llama.cpp server is not running or unreachable.")
    except requests.exceptions.HTTPError as exc:
        status = (
            exc.response.status_code if exc.response is not None else "error"
        )
        raise HTTPException(
            status_code=502,
            detail=f"Failed to query llama.cpp models (HTTP {status}).",
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unexpected error querying llama.cpp models.",
        )


@router.get("/llama_cpp/status")
async def get_llama_cpp_status():
    """Check llama.cpp runtime configuration and basic connectivity."""
    try:
        cfg = load_settings().get("llama_cpp", {})
        endpoint_raw = os.environ.get("LLAMA_CPP_BASE_URL", cfg.get("base_url", ""))
        endpoint_display = ""
        if endpoint_raw:
            try:
                endpoint_display = normalize_runtime_llama_cpp_base_url(
                    str(endpoint_raw)
                )
            except ValueError:
                endpoint_display = str(endpoint_raw).strip()

        configured = bool(str(endpoint_raw).strip())
        reachable = False
        model_count = None
        error = None
        if configured:
            try:
                models_resp = requests.get(get_llama_cpp_url("models"), timeout=5)
                models_resp.raise_for_status()
                model_count = len(models_resp.json().get("data", []))
                reachable = True
            except Exception:
                error = "failed to reach llama.cpp models endpoint"
                try:
                    health_resp = requests.get(get_llama_cpp_url("health"), timeout=5)
                    if health_resp.status_code < 500:
                        reachable = True
                except Exception:
                    pass

        return {
            "status": "success",
            "configured": configured,
            "reachable": reachable,
            "base_url": endpoint_display,
            "timeout_seconds": get_llama_cpp_timeout(),
            "model_count": model_count,
            "error": error,
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to load llama.cpp status.",
        )


@router.get("/gemini/status")
async def get_gemini_status():
    """Check if Gemini API key is configured via environment variable."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        configured = bool(api_key.strip())
        payload = {
            "status": "success",
            "configured": configured,
        }
        # Never expose key material on hosted/Railway unless explicitly enabled.
        expose_preview = os.getenv("GEMINI_STATUS_EXPOSE_KEY_PREVIEW", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if configured and (expose_preview or not _is_railway_deploy()):
            payload["key_preview"] = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else None
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/azure_openai/status")
async def get_azure_openai_status():
    """Check Azure OpenAI runtime configuration."""
    try:
        cfg = load_settings().get("azure_openai", {})
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", cfg.get("endpoint", ""))
        api_version = os.environ.get("OPENAI_API_VERSION", cfg.get("api_version", "2024-10-21"))
        key_env = cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
        api_key = os.environ.get(key_env) or os.environ.get("AZURE_OPENAI_API_KEY", "")
        chat_deployment = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", cfg.get("chat_deployment", ""))
        embedding_deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", cfg.get("embedding_deployment", ""))
        configured = bool(endpoint and api_key and chat_deployment and embedding_deployment)
        return {
            "status": "success",
            "configured": configured,
            "endpoint": endpoint,
            "api_version": api_version,
            "chat_deployment": chat_deployment,
            "embedding_deployment": embedding_deployment,
            "key_preview": f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else None,
            "missing": [
                item for item, present in {
                    "AZURE_OPENAI_ENDPOINT": bool(endpoint),
                    key_env: bool(api_key),
                    "AZURE_OPENAI_CHAT_DEPLOYMENT": bool(chat_deployment),
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": bool(embedding_deployment),
                }.items() if not present
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
