import os
import time
import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
MODELS_JSON = ROOT / "config" / "models.json"
PROFILE_YAML = ROOT / "config" / "profiles.yaml"
_genai = None
_server_error_type = None


def _load_genai():
    global _genai, _server_error_type
    if _genai is None:
        from google import genai
        from google.genai.errors import ServerError

        _genai = genai
        _server_error_type = ServerError
    return _genai


def _is_server_error(exc: Exception) -> bool:
    return _server_error_type is not None and isinstance(exc, _server_error_type)

class ModelManager:
    def __init__(self, model_name: str = None, provider: str = None):
        """
        Initialize ModelManager with flexible model specification.
        
        Args:
            model_name: The model to use. Can be:
                - A key from models.json (e.g., "gemini", "phi4")
                - An actual model name (e.g., "gemini-2.5-flash", "llama3:8b")
            provider: Optional explicit provider ("gemini", "ollama", "llama_cpp", or "azure_openai").
                      If provided, bypasses models.json lookup.
        """
        self.config = json.loads(MODELS_JSON.read_text())
        self.profile = yaml.safe_load(PROFILE_YAML.read_text())
        self._azure_fallback_provider: Optional[str] = None
        self._azure_fallback_model: Optional[str] = None

        # Load settings for local model endpoints
        try:
            from config.settings_loader import settings
            self._settings = settings
            self.ollama_base_url = settings.get("ollama", {}).get("base_url", "http://127.0.0.1:11434")
            self.llama_cpp_base_url = settings.get("llama_cpp", {}).get("base_url", "http://127.0.0.1:8080")
        except:
            self._settings = {}
            self.ollama_base_url = "http://127.0.0.1:11434"
            self.llama_cpp_base_url = "http://127.0.0.1:8080"

        # 🎯 NEW: Support explicit provider specification (from settings)
        if provider:
            self.model_type = provider
            self.text_model_key = model_name or "gemini-2.5-flash"
            
            if provider == "gemini":
                # Gemini: model_name is the actual Gemini model like "gemini-2.5-flash"
                self.model_info = {
                    "type": "gemini",
                    "model": self.text_model_key,
                    "api_key_env": "GEMINI_API_KEY"
                }
                api_key = os.getenv("GEMINI_API_KEY")
                self.client = _load_genai().Client(api_key=api_key)
            elif provider == "ollama":
                # Ollama: model_name is the Ollama model like "phi4" or "llama3:8b"
                self.model_info = {
                    "type": "ollama",
                    "model": self.text_model_key,
                    "url": {
                        "generate": f"{self.ollama_base_url}/api/generate",
                        "chat": f"{self.ollama_base_url}/api/chat"
                    }
                }
                self.client = None  # Ollama uses HTTP, no client needed
            elif provider == "llama_cpp":
                llama_cfg = self._settings.get("llama_cpp", {})
                endpoints = llama_cfg.get("endpoints", {}) if isinstance(llama_cfg.get("endpoints"), dict) else {}
                self.model_info = {
                    "type": "llama_cpp",
                    "model": self.text_model_key,
                    "url": {
                        "chat_completions": f"{self.llama_cpp_base_url}{endpoints.get('chat_completions', '/v1/chat/completions')}",
                        "embeddings": f"{self.llama_cpp_base_url}{endpoints.get('embeddings', '/v1/embeddings')}",
                    },
                }
                self.client = None  # llama.cpp uses HTTP, no client needed
            elif provider == "azure_openai":
                # Azure OpenAI: model_name is deployment name (chat deployment)
                azure_cfg = self._settings.get("azure_openai", {})
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", azure_cfg.get("endpoint", "")).rstrip("/")
                api_version = os.getenv("OPENAI_API_VERSION", azure_cfg.get("api_version", "2024-10-21"))
                key_env_name = azure_cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
                api_key = os.getenv(key_env_name) or os.getenv("AZURE_OPENAI_API_KEY", "")
                deployment = model_name or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or azure_cfg.get("chat_deployment", "")
                if not endpoint or not api_key or not deployment:
                    raise ValueError(
                        "Azure OpenAI provider requires endpoint, API key, and chat deployment "
                        "(configure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT)."
                    )
                self.model_info = {
                    "type": "azure_openai",
                    "model": deployment,
                    "endpoint": endpoint,
                    "api_version": api_version,
                    "api_key_env": key_env_name,
                }
                self.client = None
                self._azure_fallback_provider = azure_cfg.get("fallback_provider")
                self._azure_fallback_model = azure_cfg.get("fallback_model")
            else:
                raise ValueError(f"Unknown provider: {provider}")
        else:
            # 🔄 LEGACY: Lookup in models.json by key
            if model_name:
                self.text_model_key = model_name
            else:
                self.text_model_key = self.profile["llm"]["text_generation"]
            
            # Validate that the model exists in config
            if self.text_model_key not in self.config["models"]:
                available_models = list(self.config["models"].keys())
                raise ValueError(f"Model '{self.text_model_key}' not found in models.json. Available: {available_models}")
                
            self.model_info = self.config["models"][self.text_model_key]
            self.model_type = self.model_info["type"]

            # Initialize client based on model type
            if self.model_type == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                self.client = _load_genai().Client(api_key=api_key)
            elif self.model_type == "azure_openai":
                azure_cfg = self._settings.get("azure_openai", {})
                endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", azure_cfg.get("endpoint", "")).rstrip("/")
                api_version = os.getenv("OPENAI_API_VERSION", azure_cfg.get("api_version", "2024-10-21"))
                key_env_name = azure_cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
                deployment = (
                    os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
                    or azure_cfg.get("chat_deployment")
                    or self.model_info.get("model", "")
                )
                self.model_info = {
                    "type": "azure_openai",
                    "model": deployment,
                    "endpoint": endpoint,
                    "api_version": api_version,
                    "api_key_env": key_env_name,
                }
                self.client = None
                self._azure_fallback_provider = azure_cfg.get("fallback_provider")
                self._azure_fallback_model = azure_cfg.get("fallback_model")
            elif self.model_type == "llama_cpp":
                llama_cfg = self._settings.get("llama_cpp", {})
                endpoints = llama_cfg.get("endpoints", {}) if isinstance(llama_cfg.get("endpoints"), dict) else {}
                self.model_info = {
                    "type": "llama_cpp",
                    "model": self.model_info.get("model", self.text_model_key),
                    "url": {
                        "chat_completions": f"{self.llama_cpp_base_url}{endpoints.get('chat_completions', '/v1/chat/completions')}",
                        "embeddings": f"{self.llama_cpp_base_url}{endpoints.get('embeddings', '/v1/embeddings')}",
                    },
                }
                self.client = None
            # Ollama doesn't need a persistent client

        # Ollama timeout from config (used for completion stage; must allow ~240s+ per step)
        if self.model_type == "ollama":
            try:
                from config.settings_loader import get_timeout
                self._ollama_timeout_seconds = get_timeout()
            except Exception:
                self._ollama_timeout_seconds = 300
        elif self.model_type == "llama_cpp":
            try:
                from config.settings_loader import get_llama_cpp_timeout
                self._llama_cpp_timeout_seconds = get_llama_cpp_timeout()
            except Exception:
                self._llama_cpp_timeout_seconds = 300

    async def generate_text(self, prompt: str) -> str:
        if self.model_type == "gemini":
            return await self._gemini_generate(prompt)

        elif self.model_type == "ollama":
            return await self._ollama_generate(prompt)
        elif self.model_type == "llama_cpp":
            return await self._llama_cpp_generate(prompt)
        elif self.model_type == "azure_openai":
            try:
                return await self._azure_generate(prompt)
            except Exception as e:
                fallback_manager = self._build_azure_fallback_manager()
                if fallback_manager is None:
                    raise RuntimeError(f"Azure OpenAI generation failed: {str(e)}")
                print(f"[ModelManager] Azure generation failed; falling back to {fallback_manager.model_type}. Error: {e}")
                return await fallback_manager.generate_text(prompt)

        raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    async def generate_content(self, contents: list) -> str:
        """Generate content with support for text and images.
        
        Contents can contain:
        - str: Text content
        - PIL.Image: Image to process (will be base64-encoded for Ollama)
        """
        if self.model_type == "gemini":
            await self._wait_for_rate_limit()
            return await self._gemini_generate_content(contents)
        elif self.model_type == "ollama":
            # Ollama multimodal: extract text and images separately
            return await self._ollama_generate_content(contents)
        elif self.model_type == "llama_cpp":
            has_non_text = any(not isinstance(c, str) for c in contents)
            if has_non_text:
                raise RuntimeError(
                    "llama.cpp multimodal generation is not enabled in this integration. "
                    "Use an Ollama multimodal model for image content."
                )
            prompt = "\n".join(c for c in contents if isinstance(c, str))
            return await self._llama_cpp_generate(prompt)
        elif self.model_type == "azure_openai":
            # Phase 1 migration: text content is supported directly; multimodal content
            # temporarily uses fallback provider if configured.
            has_non_text = any(not isinstance(c, str) for c in contents)
            if has_non_text:
                fallback_manager = self._build_azure_fallback_manager()
                if fallback_manager is None:
                    raise RuntimeError(
                        "Azure OpenAI multimodal generation is not enabled yet. "
                        "Configure azure_openai.fallback_provider for image content."
                    )
                return await fallback_manager.generate_content(contents)
            prompt = "\n".join(c for c in contents if isinstance(c, str))
            return await self.generate_text(prompt)
        
        raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    def _build_azure_fallback_manager(self) -> Optional["ModelManager"]:
        provider = (self._azure_fallback_provider or "").strip().lower()
        if provider not in {"gemini", "ollama", "llama_cpp"}:
            return None
        model_name = self._azure_fallback_model
        if not model_name:
            agent_settings = self._settings.get("agent", {})
            if provider == "gemini":
                model_name = agent_settings.get("default_model", "gemini-2.5-flash")
            elif provider == "ollama":
                model_name = self._settings.get("models", {}).get("semantic_chunking", "gemma3:4b")
            else:
                model_name = self._settings.get("agent", {}).get("default_model", "Llama-3.2-3B-Instruct")
        try:
            return ModelManager(model_name, provider=provider)
        except Exception as e:
            print(f"[ModelManager] Failed to initialize fallback provider {provider}: {e}")
            return None

    async def _azure_generate(self, prompt: str) -> str:
        import aiohttp

        endpoint = self.model_info["endpoint"]
        deployment = self.model_info["model"]
        api_version = self.model_info.get("api_version", "2024-10-21")
        api_key_env = self.model_info.get("api_key_env", "AZURE_OPENAI_API_KEY")
        api_key = os.getenv(api_key_env) or os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Azure OpenAI API key is missing.")

        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }

        timeout = aiohttp.ClientTimeout(total=getattr(self, "_ollama_timeout_seconds", 300))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Azure OpenAI HTTP {response.status}: {body[:500]}")
                data = json.loads(body)
                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("Azure OpenAI returned no choices.")
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    return "\n".join(p for p in text_parts if p).strip()
                return (content or "").strip()

    async def _ollama_generate_content(self, contents: list) -> str:
        """Generate content with Ollama, supporting multimodal models like gemma3, llava, etc."""
        import base64
        import io
        from PIL import Image as PILImage
        
        text_parts = []
        images_base64 = []
        
        for content in contents:
            if isinstance(content, str):
                text_parts.append(content)
            elif hasattr(content, 'save'):  # PIL Image check
                # Convert PIL Image to base64
                try:
                    img = content
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'P'):
                        img = img.convert('RGB')
                    
                    # Resize if too large (Ollama has limits)
                    MAX_DIM = 1024
                    if img.width > MAX_DIM or img.height > MAX_DIM:
                        img.thumbnail((MAX_DIM, MAX_DIM), PILImage.Resampling.LANCZOS)
                    
                    # Encode to base64
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=85)
                    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                    images_base64.append(encoded)
                except Exception as e:
                    print(f"⚠️ Failed to encode image for Ollama: {e}")
        
        prompt = "\n".join(text_parts)
        
        if images_base64:
            # Use Ollama's multimodal format with images array
            return await self._ollama_generate_with_images(prompt, images_base64)
        else:
            # Text-only fallback
            return await self._ollama_generate(prompt)

    async def _ollama_generate_with_images(self, prompt: str, images: list) -> str:
        """Generate with Ollama using images (for multimodal models)."""
        try:
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=getattr(self, "_ollama_timeout_seconds", 300))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.model_info["url"]["generate"],
                    json={
                        "model": self.model_info["model"],
                        "prompt": prompt,
                        "images": images,  # Base64 encoded images
                        "stream": False
                    }
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama multimodal generation failed: {str(e)}")

    # --- Rate Limiting Helper ---
    _last_call = 0
    _lock = asyncio.Lock()

    async def _wait_for_rate_limit(self):
        """Enforce ~15 RPM limit for Gemini (4s interval)"""
        async with ModelManager._lock:
            now = time.time()
            elapsed = now - ModelManager._last_call
            if elapsed < 4.5: # 4.5s buffer for safety
                sleep_time = 4.5 - elapsed
                # print(f"[Rate Limit] Sleeping for {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            ModelManager._last_call = time.time()


    async def _gemini_generate(self, prompt: str) -> str:
        await self._wait_for_rate_limit()
        try:
            # ✅ CORRECT: Use synchronous SDK client in thread to bypass aiohttp/DNS issues common on macOS
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_info["model"],
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            if _is_server_error(e):
                # ✅ FIXED: Raise the exception instead of returning it
                raise e
            # ✅ Handle other potential errors
            raise RuntimeError(f"Gemini generation failed: {str(e)}")

    async def _gemini_generate_content(self, contents: list) -> str:
        """Generate content with support for text and images using Gemini SDK"""
        try:
            # ✅ Use synchronous SDK client in thread (text + images)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_info["model"],
                contents=contents
            )
            return response.text.strip()

        except Exception as e:
            if _is_server_error(e):
                # ✅ FIXED: Raise the exception instead of returning it
                raise e
            # ✅ Handle other potential errors
            raise RuntimeError(f"Gemini content generation failed: {str(e)}")

    async def _ollama_generate(self, prompt: str) -> str:
        try:
            # ✅ Use aiohttp for truly async requests (timeout from config for run completion)
            import aiohttp
            timeout = aiohttp.ClientTimeout(total=getattr(self, "_ollama_timeout_seconds", 300))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.model_info["url"]["generate"],
                    json = {"model": self.model_info["model"], "prompt": prompt, "stream": False}
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["response"].strip()
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

    async def _llama_cpp_generate(self, prompt: str) -> str:
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(
                total=getattr(self, "_llama_cpp_timeout_seconds", 300)
            )
            payload = {
                "model": self.model_info["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.model_info["url"]["chat_completions"],
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    choices = result.get("choices", [])
                    if not choices:
                        raise RuntimeError("llama.cpp returned no choices.")
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    if isinstance(content, list):
                        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                        return "\n".join(p for p in text_parts if p).strip()
                    return (content or "").strip()
        except Exception as e:
            raise RuntimeError(f"llama.cpp generation failed: {str(e)}")
