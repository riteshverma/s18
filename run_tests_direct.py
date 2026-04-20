"""Direct test runner that exec's each test file as a script.

Bypasses Python namespace-package scanning (no tests/__init__.py) which is
slow on Windows when PROJECT_ROOT contains large directories.
"""

from __future__ import annotations

import inspect
import json
import sys
import time
import traceback
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEST_FILES = [
    PROJECT_ROOT / "tests" / "integrations" / "test_contracts.py",
    PROJECT_ROOT / "tests" / "integrations" / "test_registry.py",
    PROJECT_ROOT / "tests" / "integrations" / "test_runs_router_contract.py",
    PROJECT_ROOT / "tests" / "core" / "test_event_bus.py",
    PROJECT_ROOT / "tests" / "core" / "test_scheduler.py",
    PROJECT_ROOT / "tests" / "core" / "test_persistence.py",
    PROJECT_ROOT / "tests" / "test_supabase_auth.py",
]

PROGRESS_FILE = PROJECT_ROOT / "test_progress.log"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(PROGRESS_FILE, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
    sys.stdout.write(line)
    sys.stdout.flush()


def _fixture_tmp_path(test_name: str) -> Path:
    base = PROJECT_ROOT / "data" / "system" / "_pytest_tmp" / test_name.replace("::", "_")
    if base.exists():
        for child in base.iterdir():
            try:
                child.unlink()
            except Exception:
                pass
    else:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _load_module(path: Path) -> types.ModuleType:
    mod_name = f"_loaded_{path.stem}_{abs(hash(str(path)))}"
    module = types.ModuleType(mod_name)
    module.__file__ = str(path)
    source = path.read_text(encoding="utf-8")
    code = compile(source, str(path), "exec")
    exec(code, module.__dict__)
    return module


def _collect_tests(module) -> list[tuple[str, object]]:
    tests = []
    for name, obj in vars(module).items():
        if not name.startswith("test_"):
            continue
        if inspect.isfunction(obj):
            tests.append((name, obj))
        elif inspect.isclass(obj):
            try:
                instance = obj()
            except TypeError:
                continue
            for method_name, method_obj in inspect.getmembers(
                instance, predicate=inspect.ismethod
            ):
                if method_name.startswith("test_"):
                    tests.append((f"{name}.{method_name}", method_obj))
    return tests


def _kwargs_for(func, test_name: str) -> dict:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {}
    kwargs = {}
    for pname in sig.parameters:
        if pname == "tmp_path":
            kwargs[pname] = _fixture_tmp_path(test_name)
    return kwargs


def run_file(path: Path) -> list[dict]:
    results: list[dict] = []
    t0 = time.time()
    label = path.relative_to(PROJECT_ROOT).as_posix()
    _log(f"LOAD {label}")
    try:
        module = _load_module(path)
    except Exception:
        _log(f"LOAD FAIL {label}")
        results.append(
            {
                "file": label,
                "test": "<load>",
                "status": "error",
                "duration_ms": int((time.time() - t0) * 1000),
                "error": traceback.format_exc(),
            }
        )
        return results
    _log(f"LOADED {label} in {int((time.time()-t0)*1000)}ms")

    for test_name, func in _collect_tests(module):
        start = time.time()
        kwargs = _kwargs_for(func, f"{label}::{test_name}")
        _log(f"RUN {label}::{test_name}")
        try:
            func(**kwargs)
            status = "passed"
            err = None
        except Exception:
            status = "failed"
            err = traceback.format_exc()
        dur = int((time.time() - start) * 1000)
        _log(f"{status.upper()} {label}::{test_name} in {dur}ms")
        results.append(
            {
                "file": label,
                "test": test_name,
                "status": status,
                "duration_ms": dur,
                "error": err,
            }
        )
    return results


def main() -> int:
    all_results: list[dict] = []
    for path in TEST_FILES:
        if not path.exists():
            _log(f"SKIP (missing) {path}")
            continue
        all_results.extend(run_file(path))

    report = {
        "total": len(all_results),
        "passed": sum(1 for r in all_results if r["status"] == "passed"),
        "failed": sum(1 for r in all_results if r["status"] == "failed"),
        "errored": sum(1 for r in all_results if r["status"] == "error"),
        "results": all_results,
    }
    (PROJECT_ROOT / "test_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n==== SUMMARY ====")
    print(f"Passed : {report['passed']} / {report['total']}")
    print(f"Failed : {report['failed']}")
    print(f"Errored: {report['errored']}")
    for r in all_results:
        if r["status"] != "passed":
            print(f"[{r['status'].upper()}] {r['file']}::{r['test']}")
            if r.get("error"):
                last_line = r["error"].strip().splitlines()[-1]
                print(f"   -> {last_line}")
    return 0 if report["failed"] == 0 and report["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
