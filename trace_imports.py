"""Isolate which import step is hanging."""
import time
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

log_path = HERE / "import_trace.log"
log_path.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
    print(line, end="", flush=True)


def step(label: str, func):
    t = time.time()
    log(f"START {label}")
    try:
        func()
        log(f"OK {label} in {int((time.time()-t)*1000)}ms")
    except Exception as e:
        log(f"FAIL {label}: {e!r}")


step("pydantic", lambda: __import__("pydantic"))
step("prometheus_client", lambda: __import__("prometheus_client"))
step("integrations.contracts", lambda: __import__("integrations.contracts", fromlist=["X"]))
step("integrations.profiles", lambda: __import__("integrations.profiles", fromlist=["X"]))
step("integrations.adapters.default", lambda: __import__("integrations.adapters.default", fromlist=["X"]))
step("integrations.adapters.wiseai", lambda: __import__("integrations.adapters.wiseai", fromlist=["X"]))
step("integrations.registry", lambda: __import__("integrations.registry", fromlist=["X"]))
step("integrations (pkg)", lambda: __import__("integrations"))
log("DONE")
