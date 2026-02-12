# web_tools_async.py
import asyncio
import traceback
import time
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import random
from pathlib import Path
import sys
import os

# MCP Protocol Safety: Redirect print to stderr
def print(*args, **kwargs):
    sys.stderr.write(" ".join(map(str, args)) + "\n")
    sys.stderr.flush()

DIFFICULT_WEBSITES_PATH = Path(__file__).parent / "difficult_websites.txt"

# --- Configurable timeouts (env vars) ---
def _config_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or not str(val).strip():
        return default
    try:
        return int(val)
    except ValueError:
        return default

EXTRACT_FAST_TIMEOUT = _config_int("S18_EXTRACT_FAST_TIMEOUT", 5)
EXTRACT_PLAYWRIGHT_GOTO = _config_int("S18_EXTRACT_PLAYWRIGHT_GOTO_MS", 15000)
EXTRACT_PLAYWRIGHT_WAIT = _config_int("S18_EXTRACT_PLAYWRIGHT_WAIT_MS", 15000)
EXTRACT_CACHE_TTL_SEC = _config_int("S18_EXTRACT_CACHE_TTL_SEC", 300)
EXTRACT_RETRIES = _config_int("S18_EXTRACT_RETRIES", 2)
EXTRACT_BACKOFF_BASE = _config_int("S18_EXTRACT_BACKOFF_BASE_SEC", 1)

# --- Short-TTL cache for extract results ---
_extract_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = asyncio.Lock()

async def _retry_with_backoff(async_func, max_retries: int = EXTRACT_RETRIES, base_delay: float = EXTRACT_BACKOFF_BASE):
    """Retry an async function with exponential backoff."""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await async_func()
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
                await asyncio.sleep(delay)
    raise last_exc

def get_random_headers():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/537.36 Chrome/113.0.5672.92 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 Version/16.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (Linux; Android 13; Pixel 6) AppleWebKit/537.36 Chrome/117.0.5938.132 Mobile Safari/537.36",
        "Mozilla/5.0 (Linux; Android 13; SAMSUNG SM-G998B) AppleWebKit/537.36 Chrome/92.0.4515.159 Mobile Safari/537.36 SamsungBrowser/15.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile Safari/604.1",
        "Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 Version/16.6 Mobile Safari/604.1"
    ]
    return {"User-Agent": random.choice(user_agents)}


def is_difficult_website(url: str) -> bool:
    if not DIFFICULT_WEBSITES_PATH.exists():
        return False
    try:
        with open(DIFFICULT_WEBSITES_PATH, "r", encoding="utf-8") as f:
            difficult_sites = [line.strip().lower() for line in f if line.strip()]
        return any(domain in url.lower() for domain in difficult_sites)
    except Exception as e:
        print(f"⚠️ Failed to read difficult_websites.txt: {e}")
        return False

# Make sure these utilities exist
def ascii_only(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode()

def choose_best_text(visible, main, trafilatura_):
    # Simple heuristic: prefer main if long, fallback otherwise
    scores = {
        "visible": len(visible.strip()),
        "main": len(main.strip()),
        "trafilatura": len(trafilatura_.strip())
    }
    best = max(scores, key=scores.get)
    return {
        "visible": visible,
        "main": main,
        "trafilatura": trafilatura_
    }[best], best

async def web_tool_playwright(url: str, max_total_wait: int = None) -> dict:
    goto_ms = max_total_wait * 1000 if max_total_wait else EXTRACT_PLAYWRIGHT_GOTO
    wait_ms = EXTRACT_PLAYWRIGHT_WAIT
    result = {"url": url}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)  # changed to headless=True for stability
            page = await browser.new_page()

            await page.goto(url, wait_until="domcontentloaded", timeout=goto_ms)

            # Wait until the page body has significant content (i.e., text is non-trivial)
            try:
                await page.wait_for_function(
                    """() => {
                        const body = document.querySelector('body');
                        return body && (body.innerText || "").length > 1000;
                    }""",
                    timeout=wait_ms
                )
            except Exception as e:
                print("⚠️ Generic wait failed:", e)

            # Optional light sleep for residual JS rendering
            await asyncio.sleep(5)

            try:
                await page.evaluate("""() => {
                    window.stop();
                    document.querySelectorAll('script').forEach(s => s.remove());
                }""")
            except Exception as e:
                print("⚠️ JS stop failed:", e)

            html = await page.content()
            visible_text = await page.inner_text("body")
            title = await page.title()
            await browser.close()

            # Run parsing in background to free browser early
            try:
                main_text = await asyncio.to_thread(lambda: BeautifulSoup(Document(html).summary(), "html.parser").get_text(separator="\n", strip=True))
            except Exception as e:
                print("⚠️ Readability failed:", e)
                main_text = ""

            try:
                trafilatura_text = await asyncio.to_thread(lambda: trafilatura.extract(html) or "")
            except Exception as e:
                print("⚠️ Trafilatura failed:", e)
                trafilatura_text = ""

            best_text, source = choose_best_text(visible_text, main_text, trafilatura_text)

            result.update({
                "title": title,
                "html": html,
                "text": visible_text,
                "main_text": main_text,
                "trafilatura_text": trafilatura_text,
                "best_text": ascii_only(best_text),
                "best_text_source": source
            })

    except PlaywrightTimeoutError:
        result.update({
            "title": "[timeout: goto]",
            "html": "",
            "text": "[timed out]",
            "main_text": "[no HTML extracted]",
            "trafilatura_text": "",
            "best_text": "[no text]",
            "best_text_source": "timeout"
        })

    except Exception as e:
        traceback.print_exc()
        result.update({
            "title": "[error]",
            "html": "",
            "text": f"[error: {e}]",
            "main_text": "[no HTML extracted]",
            "trafilatura_text": "",
            "best_text": "[no text]",
            "best_text_source": "error"
        })

    return result

import httpx

# --- Shared HTTP client (connection reuse) ---
_shared_http_client: httpx.AsyncClient | None = None

def _get_http_client() -> httpx.AsyncClient:
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            timeout=EXTRACT_FAST_TIMEOUT,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _shared_http_client

async def _smart_web_extract_impl(url: str, timeout: int = None) -> dict:
    """Internal implementation without cache."""
    use_timeout = timeout if timeout is not None else EXTRACT_FAST_TIMEOUT
    headers = get_random_headers()

    try:
        if is_difficult_website(url):
            print(f"Detected difficult site ({url}) → skipping fast scrape")
            return await web_tool_playwright(url)

        client = _get_http_client()
        response = await client.get(url, headers=headers, timeout=use_timeout)
        html = response.content.decode("utf-8", errors="replace")

        doc = Document(html)
        main_html = doc.summary()
        main_text = BeautifulSoup(main_html, "html.parser").get_text(separator="\n", strip=True)
        visible_text = BeautifulSoup(html, "html.parser").get_text(separator="\n", strip=True)
        trafilatura_text = trafilatura.extract(html)
        best_text, best_source = choose_best_text(visible_text, main_text, trafilatura_text)

        if len(best_text) >= 300:
            return {
                "url": url,
                "title": Document(html).short_title(),
                "html": html,
                "text": visible_text,
                "main_text": main_text,
                "trafilatura_text": trafilatura_text,
                "best_text": ascii_only(best_text),
                "best_text_source": best_source,
            }

        print("Fast scrape too small, falling back...")

    except Exception as e:
        print("Fast scrape failed:", e)

    # Fallback
    return await web_tool_playwright(url)

async def smart_web_extract(url: str, timeout: int = None) -> dict:
    """Extract text from URL with cache, retries, and shared HTTP client."""
    # Check cache first
    async with _cache_lock:
        now = time.time()
        if url in _extract_cache:
            cached_at, cached_result = _extract_cache[url]
            if (now - cached_at) < EXTRACT_CACHE_TTL_SEC:
                return cached_result.copy()
            del _extract_cache[url]

    effective_timeout = timeout if timeout is not None else EXTRACT_FAST_TIMEOUT

    async def do_extract():
        result = await _smart_web_extract_impl(url, timeout=effective_timeout)
        async with _cache_lock:
            _extract_cache[url] = (time.time(), result.copy())
        return result

    return await _retry_with_backoff(do_extract)


if __name__ == "__main__":
    print("starting scrape subprocess...")
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python web_tool_playwright_async.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    print("🚀 Trying smart scrape first...")
    result = asyncio.run(smart_web_extract(url))
    print(json.dumps(result, ensure_ascii=False))
