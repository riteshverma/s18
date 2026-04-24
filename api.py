import sys
import os
import asyncio
import subprocess
from pathlib import Path

# Windows: ProactorEventLoop required for asyncio subprocess (git clone, uv run)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest  # type: ignore[reportMissingImports]

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.loop import AgentLoop4
from core.scheduler import scheduler_service
from core.persistence import persistence_manager
from core.graph_adapter import nx_to_reactflow
from memory.context import ExecutionContextManager
from remme.utils import get_embedding
from config.settings_loader import (
    settings,
    save_settings,
    reset_settings,
    reload_settings,
    get_mcp_startup_timeout,
    get_run_poll_timeout,
)
from core.supabase_auth import is_auth_enabled
from core.supabase_logging import is_logging_enabled
from core.supabase_config import get_supabase_config


# Import shared state
from shared.state import (
    active_loops,
    get_multi_mcp,
    get_remme_store,
    get_remme_extractor,
    PROJECT_ROOT,
)
from routers.remme import background_smart_scan  # Needed for lifespan startup
from core.prometheus_metrics import (
    API_REQUEST_LATENCY_MS,
    API_REQUESTS_SUCCESS_TOTAL,
    API_REQUESTS_TOTAL,
    elapsed_ms,
    normalize_status_class,
    now_ms,
    route_template,
)

from contextlib import asynccontextmanager

# Get shared instances
multi_mcp = get_multi_mcp()
remme_store = get_remme_store()
remme_extractor = get_remme_extractor()
_mcp_start_task: Optional[asyncio.Task] = None


async def _start_mcp_with_timeout(timeout_seconds: Optional[float] = None) -> None:
    """
    Start MCP servers without blocking API readiness for long boot phases.
    If startup exceeds timeout, continue startup in background.
    """
    global _mcp_start_task
    timeout_seconds = timeout_seconds or get_mcp_startup_timeout()
    _mcp_start_task = asyncio.create_task(multi_mcp.start())
    try:
        await asyncio.wait_for(asyncio.shield(_mcp_start_task), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        if multi_mcp.is_strict_mode():
            raise RuntimeError(
                f"MCP strict mode startup timed out after {timeout_seconds}s"
            )
        print(f"⚠️ MCP startup exceeded {timeout_seconds}s; continuing in background.")
    except Exception:
        # Re-raise non-timeout failures so startup still surfaces real errors.
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API starting up...")
    scheduler_service.initialize()
    scheduler_service.register_morning_briefing()
    persistence_manager.load_snapshot()
    await _start_mcp_with_timeout()
    
    # Check git
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        print("Git found.")
    except Exception:
        print("WARNING: Git NOT found. GitHub explorer features will fail.")
    
    # 🧠 Start Smart Sync in background
    asyncio.create_task(background_smart_scan())
    
    yield

    print("API shutting down...")
    persistence_manager.save_snapshot()
    global _mcp_start_task
    try:
        if _mcp_start_task and not _mcp_start_task.done():
            _mcp_start_task.cancel()
            try:
                await _mcp_start_task
            except Exception:
                pass
        await asyncio.wait_for(multi_mcp.stop(), timeout=3.0)
    except asyncio.CancelledError:
        pass
    except (RuntimeError, ExceptionGroup, BaseExceptionGroup) as e:
        if "cancel scope" in str(e).lower() or "TaskGroup" in str(type(e).__name__):
            pass  # MCP stdio teardown noise on Ctrl+C
        else:
            print(f"⚠️ Shutdown warning: {e}")
    except Exception as e:
        print(f"⚠️ Shutdown warning: {e}")

app = FastAPI(lifespan=lifespan)

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "app://."], # Explicitly allow frontend
    allow_origin_regex=r"http://localhost:(517\d|5555)", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    start_ms = now_ms()
    response = await call_next(request)
    route = route_template(request.url.path, request.scope.get("route"))
    method = request.method
    status_code = response.status_code

    API_REQUESTS_TOTAL.labels(
        method=method,
        route=route,
        status_class=normalize_status_class(status_code),
    ).inc()
    API_REQUEST_LATENCY_MS.labels(method=method, route=route).observe(elapsed_ms(start_ms))
    if status_code < 400:
        API_REQUESTS_SUCCESS_TOTAL.labels(method=method, route=route).inc()
    return response


# Global State is now managed in shared/state.py
# active_loops, multi_mcp, remme_store, remme_extractor are imported from there

# === Import and Include Routers ===
from routers import runs as runs_router
from routers import rag as rag_router
from routers import remme as remme_router
from routers import apps as apps_router
from routers import settings as settings_router
from routers import explorer as explorer_router
from routers import mcp as mcp_router

# MCP router mounted first with explicit empty prefix so /mcp/* routes are reachable
app.include_router(mcp_router.router, tags=["MCP"])
app.include_router(runs_router.router)
app.include_router(rag_router.router)
app.include_router(remme_router.router)
app.include_router(apps_router.router)
app.include_router(settings_router.router)
app.include_router(explorer_router.router)
from routers import prompts as prompts_router
from routers import news as news_router
from routers import git as git_router
app.include_router(prompts_router.router)
app.include_router(news_router.router)
app.include_router(git_router.router)

from routers import chat as chat_router
app.include_router(chat_router.router)
from routers import agent as agent_router
app.include_router(agent_router.router)
from routers import ide_agent as ide_agent_router
app.include_router(ide_agent_router.router)
from routers import metrics as metrics_router
app.include_router(metrics_router.router)
from routers import python_tools
app.include_router(python_tools.router)
from routers import tests as tests_router
app.include_router(tests_router.router)
# Chat router included
from routers import inbox
app.include_router(inbox.router)
from routers import cron
app.include_router(cron.router)
from routers import stream
app.include_router(stream.router)
from routers import agui as agui_router
app.include_router(agui_router.router)
from routers import skills
app.include_router(skills.router)





@app.get("/health")
async def health_check():
    mcp_health = multi_mcp.get_health_status()
    return {
        "status": "ok",
        "version": "1.0.0",
        "mcp_ready": mcp_health["mcp_ready"],
        "mcp_mode": mcp_health["mode"],
        "mcp_start_completed": mcp_health["start_completed"],
        "mcp_connected_servers": mcp_health["connected_servers"],
        "mcp_required_servers": mcp_health["required_servers"],
        "run_poll_timeout_seconds": get_run_poll_timeout(),
    }


@app.get("/health/auth")
async def health_auth():
    """Quick auth/logging diagnostics without exposing secrets."""
    cfg = get_supabase_config()

    return {
        "status": "ok",
        "auth_enabled": is_auth_enabled(),
        "supabase_logging_enabled": is_logging_enabled(),
        "supabase_url_configured": bool(cfg.get("url")),
        "supabase_jwt_audience_configured": bool(cfg.get("jwt_audience")),
        "supabase_anon_key_configured": bool(cfg.get("anon_key")),
        "supabase_service_role_key_configured": bool(cfg.get("service_role_key")),
        "request_table": cfg.get("request_table", "ehr_request_log"),
        "result_table": cfg.get("result_table", "ehr_clinical_result"),
    }


@app.get("/metrics/prometheus")
async def prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    # Enable reload=True for development if needed, but here we'll just keep it simple
    # or actually enable it to avoid these restart issues.
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True, timeout_keep_alive=75)
