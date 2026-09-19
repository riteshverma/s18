import ast
import asyncio
import time
import builtins
import textwrap
import re
import os
import json
import inspect
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
import sys
import traceback
from core.utils import log_step, log_error, log_json_block
import io
import contextlib

# MCP Protocol Safety: Redirect print to stderr
def print(*args, **kwargs):
    sys.stderr.write(" ".join(map(str, args)) + "\n")
    sys.stderr.flush()

# from agent.agentSession import ExecutionSnapshot

ALLOWED_MODULES = {
    "math", "random", "re", "datetime", "time", "collections", "itertools",
    "statistics", "string", "functools", "operator", "json", "pprint", "copy",
    "typing", "uuid", "hashlib", "base64", "hmac", "struct", "decimal", "fractions"
}

SAFE_BUILTINS = [
    # Core types and structure
    "bool", "int", "float", "str", "list", "dict", "set", "tuple", "complex",
    
    # Iteration and collection helpers
    "range", "enumerate", "zip", "map", "filter", "reversed", "next",
    
    # Logic and math
    "abs", "round", "divmod", "pow", "sum", "min", "max", "all", "any",
    
    # String and character
    "ord", "chr", "len", "sorted",
    
    # Type inspection
    "isinstance", "issubclass", "type", "id",
    
    # Functional
    "callable", "hash", "format",
    
    # Import-related
    "__import__",

    # Output and utility
    "print", "locals", "globals", "repr",
    "Exception", "True", "False", "None", "open"
]

MAX_FUNCTIONS = 20
TIMEOUT_PER_FUNCTION = 50

REPO_ROOT = Path(__file__).parent.parent
DATA_TMP_DIR = REPO_ROOT / "data" / "tmp"
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 20
# Sentinel prefix on the child's final stdout line carrying the JSON result payload.
SUBPROCESS_RESULT_SENTINEL = "__SANDBOX_RESULT__"
SUBPROCESS_MEMORY_LIMIT_MB = 512

# ===== SECURITY: BLOCKED PATTERNS =====
BLOCKED_PATTERNS = [
    # File system attacks
    (r"rm\s+-rf", "Recursive file deletion"),
    (r"shutil\.rmtree", "Directory deletion"),
    (r"os\.remove\(", "File deletion"),
    (r"os\.unlink\(", "File deletion"),
    
    # SQL injection
    (r"DROP\s+TABLE", "SQL DROP TABLE"),
    (r"DELETE\s+FROM\s+\w+\s*;?\s*$", "SQL DELETE without WHERE"),
    (r"TRUNCATE\s+TABLE", "SQL TRUNCATE"),
    
    # Code execution
    (r"os\.system\(", "Shell command execution"),
    (r"subprocess\.", "Subprocess execution"),
    (r"eval\s*\(", "Eval execution"),
    (r"exec\s*\(", "Exec execution"),
    (r"__import__\s*\(\s*['\"]os", "Dynamic os import"),
    
    # Network access (if not explicitly allowed)
    (r"socket\.", "Raw socket access"),
    
    # Sensitive file access
    (r"open\s*\(\s*['\"]\/etc\/", "System file access"),
    (r"open\s*\(\s*['\"]\/proc\/", "Proc file access"),
    
    # Crypto mining / resource abuse
    (r"while\s+True\s*:", "Infinite loop pattern"),
    (r"for\s+_\s+in\s+iter\s*\(\s*int\s*,\s*1\s*\)", "Infinite iterator"),
]

SECURITY_LOG_PATH = Path(__file__).parent.parent / "data" / "security_logs"


def check_code_safety(code: str) -> tuple[bool, list[dict]]:
    """
    Check code for dangerous patterns.
    
    Returns:
        (is_safe, violations)
        where violations is a list of {"pattern": str, "description": str, "match": str}
    """
    violations = []
    
    for pattern, description in BLOCKED_PATTERNS:
        matches = re.finditer(pattern, code, re.IGNORECASE)
        for match in matches:
            violations.append({
                "pattern": pattern,
                "description": description,
                "match": match.group(),
                "position": match.start()
            })
    
    return len(violations) == 0, violations


def log_security_event(event: dict):
    """
    Log a security event to file.
    
    Creates a daily log file with all blocked attempts.
    Example output in data/security_logs/2026-01-15.jsonl
    """
    try:
        SECURITY_LOG_PATH.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = SECURITY_LOG_PATH / f"{today}.jsonl"
        
        event["timestamp"] = datetime.now().isoformat()
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        
        # Also log to console for immediate visibility
        log_error(f"🚨 SECURITY: {event.get('action', 'EVENT')} - {event.get('violation', 'Unknown')}")
    except Exception as e:
        log_error(f"Failed to log security event: {e}")

class KeywordStripper(ast.NodeTransformer):
    """Rewrite all function calls to remove keyword args and keep only values as positional."""
    def visit_Call(self, node):
        self.generic_visit(node)
        if node.keywords:
            # Convert all keyword arguments into positional args (discard names)
            for kw in node.keywords:
                node.args.append(kw.value)
            node.keywords = []
        return node


# ───────────────────────────────────────────────────────────────
# AST TRANSFORMER: auto-await known async MCP tools
# ───────────────────────────────────────────────────────────────
class AwaitTransformer(ast.NodeTransformer):
    def __init__(self, async_funcs):
        self.async_funcs = async_funcs

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.async_funcs:
            return ast.Await(value=node)
        return node



def fix_unterminated_triple_quotes(code: str) -> str:
    import re
    triple_quotes = re.findall(r'''"""''', code)
    if len(triple_quotes) % 2 != 0:
        log_error("Fixing unterminated triple-quoted string...", symbol="⚠️ ")
        return code + '\n"""'
    return code


def build_safe_globals(mcp_funcs: dict, multi_mcp=None, session_id: str = None) -> dict:
    safe_globals = {
        "__builtins__": {
            k: getattr(builtins, k) for k in SAFE_BUILTINS
        },
        **mcp_funcs,
    }

    for module in ALLOWED_MODULES:
        safe_globals[module] = __import__(module)

    safe_globals["final_answer"] = lambda x: safe_globals.setdefault("result_holder", x)

    if session_id:
        safe_globals.update(load_session_vars(session_id))
        
    # Inject DATA_DIR so agents know where to look
    safe_globals["DATA_DIR"] = str(Path(__file__).parent.parent / "data")

    if multi_mcp:
        async def parallel(*tool_calls):
            coros = [multi_mcp.function_wrapper(tool_name, *args) for tool_name, *args in tool_calls]
            return await asyncio.gather(*coros)
        safe_globals["parallel"] = parallel

    # Allow both direct access (`urls`) and schema-style (`globals_schema.get("urls", "")`)
    safe_globals["globals_schema"] = {
        k: v for k, v in safe_globals.items() if k not in {"__builtins__", "final_answer", "parallel"}
    }

    return safe_globals


def save_session_vars(session_id: str, variables: dict):
    os.makedirs("action/sandbox_state", exist_ok=True)
    path = f"action/sandbox_state/{session_id}.json"

    # Load existing vars if any
    try:
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {}

    # Merge
    merged = {**existing, **variables}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def load_session_vars(session_id: str) -> dict:
    try:
        with open(f"action/sandbox_state/{session_id}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def count_function_calls(code: str) -> int:
    tree = ast.parse(code)
    return sum(isinstance(node, ast.Call) for node in ast.walk(tree))


def serialize_result_value(v):
    """Serialize a value to JSON-compatible format.

    CRITICAL: Handles MCP tool results that may return:
    - Raw JSON-serializable values
    - Objects with .content attribute (MCP responses)
    - String representations of Python lists/dicts

    The last case caused a bug where urls were stored as strings like "['url1', 'url2']"
    and then iterated character-by-character in the next iteration.

    Module level so the sandbox child process can embed it verbatim (via
    inspect.getsource) and apply the exact same serialization.
    """
    # Already JSON-compatible primitives
    if isinstance(v, (int, float, bool, type(None))):
        return v

    # Lists and dicts - recursively serialize contents
    if isinstance(v, list):
        return [serialize_result_value(item) for item in v]
    if isinstance(v, dict):
        return {k: serialize_result_value(val) for k, val in v.items()}

    # Strings - check if they're actually serialized lists/dicts
    if isinstance(v, str):
        stripped = v.strip()
        # Check if it looks like a Python/JSON list or dict
        if (stripped.startswith('[') and stripped.endswith(']')) or \
           (stripped.startswith('{') and stripped.endswith('}')):
            # Try JSON first
            try:
                parsed = json.loads(stripped)
                return serialize_result_value(parsed)  # Recursively process
            except (json.JSONDecodeError, TypeError):
                pass

            # Try Python literal (handles single quotes, True/False/None)
            try:
                parsed = ast.literal_eval(stripped)
                return serialize_result_value(parsed)  # Recursively process
            except (ValueError, SyntaxError):
                pass

        # It's just a regular string
        return v

    # MCP ActionResultOutput with success/content/error attributes
    if hasattr(v, "success") and hasattr(v, "content") and hasattr(v, "error"):
        if not v.success:
            return f"Error executing tool: {v.error}"
        return serialize_result_value(v.content) if v.content else "Success"

    # MCP response with .content list (common pattern)
    if hasattr(v, "content") and isinstance(v.content, list):
        text_content = "\n".join(x.text for x in v.content if hasattr(x, "text"))

        # Try to parse as structured data
        stripped = text_content.strip()
        if (stripped.startswith('[') and stripped.endswith(']')) or \
           (stripped.startswith('{') and stripped.endswith('}')):
            # Try JSON first
            try:
                parsed = json.loads(stripped)
                return serialize_result_value(parsed)
            except (json.JSONDecodeError, TypeError):
                pass

            # Try Python literal
            try:
                parsed = ast.literal_eval(stripped)
                return serialize_result_value(parsed)
            except (ValueError, SyntaxError):
                pass

        return text_content

    # Fallback: convert to string
    return str(v)


def _result_error_message(result_value, returned=None) -> str | None:
    """Detect MCP tool failures surfaced as error strings or success=False objects.

    `returned` (raw objects) only exists on the in-process path; the subprocess
    path reports plain JSON data and only needs the string checks.
    """
    if not isinstance(result_value, dict):
        return None
    for v in result_value.values():
        if isinstance(v, str) and (
            v.lower().startswith("error executing tool") or
            v.lower().startswith("error:") or
            "failed" in v.lower()
        ):
            return v
    if returned:
        for k, v in returned.items():
            if hasattr(v, "success") and not v.success:
                return v.error if hasattr(v, "error") and v.error else f"Tool {k} failed"
    return None


def _get_sandbox_timeout_seconds() -> float:
    """Sandbox subprocess wall-clock timeout, configurable via settings (sandbox.timeout_seconds).

    Imported lazily: loading full settings must never break the sandbox module
    import (the MCP sandbox server loads this file at startup).
    """
    try:
        from config.settings_loader import get_sandbox_timeout_seconds as _configured
        return _configured()
    except Exception:
        return DEFAULT_SANDBOX_TIMEOUT_SECONDS


def _build_subprocess_script(user_module_source: str, session_vars: dict) -> str:
    """Assemble the isolated child script: trusted bootstrap + transformed user code.

    The child runs `python -I` (repo not on sys.path), so the bootstrap must be
    self-contained. Mirrors build_safe_globals for the multi_mcp=None case:
    restricted builtins, allowed modules, session vars, DATA_DIR, final_answer
    and globals_schema.
    """
    module_imports = "\n".join(f"import {name}" for name in sorted(ALLOWED_MODULES))
    parts = [
        "# Auto-generated sandbox child script (safe to delete).",
        "# Trusted bootstrap; the untrusted user module is appended further below.",
        "import ast",
        "import asyncio as _asyncio",
        "import builtins as _builtins_module",
        "import json",
        "import sys",
        "import traceback",
        "",
        f"_SENTINEL = {SUBPROCESS_RESULT_SENTINEL!r}",
        f"_SAFE_BUILTIN_NAMES = {sorted(SAFE_BUILTINS)!r}",
        f"_SAFE_ALLOWED_MODULE_NAMES = {sorted(ALLOWED_MODULES)!r}",
        "__builtins__ = {k: getattr(_builtins_module, k) for k in _SAFE_BUILTIN_NAMES}",
        "",
        module_imports,
        "",
        f"DATA_DIR = {str(REPO_ROOT / 'data')!r}",
        "",
        inspect.getsource(serialize_result_value),
        "",
        # Mirrors the setdefault semantics of build_safe_globals' final_answer.
        "def final_answer(x):",
        '    _g = globals()',
        '    if "result_holder" not in _g:',
        '        _g["result_holder"] = x',
        '    return _g["result_holder"]',
        "",
        f"_SESSION_VARS = {session_vars!r}",
        "globals().update(_SESSION_VARS)",
        # Session vars intentionally override module names, matching in-process ordering.
        "globals_schema = dict(_SESSION_VARS)",
        'globals_schema["DATA_DIR"] = DATA_DIR',
        "for _module_name in _SAFE_ALLOWED_MODULE_NAMES:",
        "    if _module_name not in globals_schema:",
        "        globals_schema[_module_name] = globals()[_module_name]",
        "",
        "def _emit_result(payload):",
        '    sys.stdout.write("\\n" + _SENTINEL + json.dumps(payload, default=str) + "\\n")',
        "    sys.stdout.flush()",
        "",
        user_module_source,
        "",
        "async def _sandbox_main():",
        "    try:",
        "        returned = await __main()",
        "        if isinstance(returned, dict):",
        "            result_value = {k: serialize_result_value(v) for k, v in returned.items()}",
        "        else:",
        '            result_value = {"result": serialize_result_value(returned)}',
        '        _emit_result({"status": "ok", "result": result_value})',
        "    except Exception as exc:",
        "        _emit_result({",
        '            "status": "error",',
        '            "error": f"{type(exc).__name__}: {str(exc)}",',
        '            "traceback": traceback.format_exc(),',
        "        })",
        "",
        "_asyncio.run(_sandbox_main())",
    ]
    return "\n".join(parts) + "\n"


def _communicate_with_timeout(proc: subprocess.Popen, timeout_seconds: float):
    """Blocking communicate(); kills the child once the wall-clock budget is spent.

    Returns (stdout_bytes, stderr_bytes, timed_out).
    """
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return stdout, stderr, False
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = b"", b""
        return stdout, stderr, True


async def _run_user_code_in_subprocess(
    user_module_source: str, session_id: str, start_time: float, start_timestamp: str
) -> dict:
    """Default hardened execution path: fresh `python -I` interpreter.

    AST inspection is bypassable (getattr indirection, builtins tricks), so
    code runs in a disposable process instead of the agent process. On POSIX
    the child additionally gets CPU/address-space rlimits; on Windows we rely
    on the wall-clock timeout kill. The child reports its result as JSON on a
    sentinel-marked final stdout line so the run_user_code return contract
    stays identical for callers.
    """
    timeout_seconds = _get_sandbox_timeout_seconds()
    script_path = None
    try:
        DATA_TMP_DIR.mkdir(parents=True, exist_ok=True)
        fd, script_path = tempfile.mkstemp(
            prefix="sandbox_exec_", suffix=".py", dir=str(DATA_TMP_DIR)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as script_file:
            script_file.write(
                _build_subprocess_script(user_module_source, load_session_vars(session_id))
            )

        popen_kwargs = {}
        if os.name == "posix":
            cpu_limit = int(timeout_seconds) + 1
            memory_bytes = SUBPROCESS_MEMORY_LIMIT_MB * 1024 * 1024

            def _limit_sandbox_resources():
                # preexec_fn hook (POSIX only): `resource` does not exist on Windows.
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            popen_kwargs["preexec_fn"] = _limit_sandbox_resources

        proc = subprocess.Popen(
            [sys.executable, "-I", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=str(REPO_ROOT),
            **popen_kwargs,
        )
        try:
            stdout_bytes, stderr_bytes, timed_out = await asyncio.to_thread(
                _communicate_with_timeout, proc, timeout_seconds
            )
        finally:
            if proc.poll() is None:
                proc.kill()
    finally:
        if script_path:
            Path(script_path).unlink(missing_ok=True)

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    if timed_out:
        return {
            "status": "error",
            "error": f"Execution timed out after {timeout_seconds:g} seconds",
            "logs": stdout_text + stderr_text,
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    payload = None
    for line in reversed(stdout_text.splitlines()):
        if line.startswith(SUBPROCESS_RESULT_SENTINEL):
            try:
                payload = json.loads(line[len(SUBPROCESS_RESULT_SENTINEL):])
            except json.JSONDecodeError:
                payload = None
            break

    if not isinstance(payload, dict) or "status" not in payload:
        return {
            "status": "error",
            "error": f"Sandbox subprocess crashed before reporting a result (exit code {proc.returncode})",
            "traceback": stderr_text[-2000:],
            "logs": (stdout_text + stderr_text)[-4000:],
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    if payload.get("status") == "error":
        return {
            "status": "error",
            "error": payload.get("error", "Unknown sandbox error"),
            "traceback": payload.get("traceback", ""),
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    result_value = payload.get("result")
    if not isinstance(result_value, dict):
        result_value = {"result": result_value}

    error_msg = _result_error_message(result_value)
    if error_msg:
        return {
            "status": "error",
            "error": error_msg,
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    log_json_block("Executor result", result_value)

    save_session_vars(session_id, result_value)

    # Drop the sentinel payload line so logs only contain user code output.
    clean_lines = [
        line for line in stdout_text.splitlines()
        if not line.startswith(SUBPROCESS_RESULT_SENTINEL)
    ]
    clean_stdout = "\n".join(clean_lines)
    if clean_stdout and stderr_text:
        logs = clean_stdout + "\n" + stderr_text
    else:
        logs = clean_stdout or stderr_text

    return {
        "status": "success",
        "result": result_value,
        "raw": result_value,
        "logs": logs,
        "execution_time": start_timestamp,
        "total_time": str(round(time.perf_counter() - start_time, 3))
    }


def make_tool_proxy(tool_name: str, mcp):
    async def _tool_fn(*args):
        return await mcp.function_wrapper(tool_name, *args)
    return _tool_fn

async def run_user_code(code: str, multi_mcp, session_id: str = "default_session") -> dict:
    start_time = time.perf_counter()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ===== SAFETY CHECK BEFORE EXECUTION =====
    is_safe, violations = check_code_safety(code)
    
    if not is_safe:
        # Log the blocked attempt
        log_security_event({
            "session_id": session_id,
            "action": "BLOCKED",
            "violation": violations[0]["description"],
            "pattern_matched": violations[0]["pattern"],
            "code_snippet": code[:500],  # First 500 chars only
            "all_violations": [v["description"] for v in violations]
        })
        
        return {
            "status": "blocked",
            "error": f"Security violation: {violations[0]['description']}",
            "violations": [v["description"] for v in violations],
            "blocked_pattern": violations[0]["match"],
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    def is_json_serializable(value):
        return isinstance(value, (str, int, float, bool, type(None), list, dict))

    try:
        func_count = count_function_calls(code)
        if func_count > MAX_FUNCTIONS:
            return {
                "status": "error",
                "error": f"Too many functions ({func_count} > {MAX_FUNCTIONS})",
                "execution_time": start_timestamp,
                "total_time": str(round(time.perf_counter() - start_time, 3))
            }

        tool_funcs = {
            tool.name: make_tool_proxy(tool.name, multi_mcp)
            for tool in multi_mcp.get_all_tools()
        } if multi_mcp is not None else {}

        log_step(f"[CODE:]: {code}", symbol="🐍")

        cleaned_code = fix_unterminated_triple_quotes(textwrap.dedent(code.strip()))
        tree = ast.parse(cleaned_code)

        # ─── AST Transformations ─────────────────────────────────────
        tree = KeywordStripper().visit(tree)
        tree = AwaitTransformer(set(tool_funcs)).visit(tree)

        # Rewrite return <varname> → return {"varname": varname}
        new_body = []
        return_found = False
        for node in tree.body:
            if isinstance(node, ast.Return):
                return_found = True
                if isinstance(node.value, ast.Name):
                    varname = node.value.id
                    new_body.append(
                        ast.Return(
                            value=ast.Dict(
                                keys=[ast.Constant(value=varname)],
                                values=[ast.Name(id=varname, ctx=ast.Load())]
                            )
                        )
                    )
                else:
                    new_body.append(node)
            else:
                new_body.append(node)

        # If return is missing but 'result' exists, add `return result`
        has_result_var = any(  # noqa: F841 -- return-injection not wired up yet; kept for pending feature
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "result" for t in node.targets)
            for node in new_body
        )
        result_vars = {
            node.targets[0].id
            for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        }

        if not return_found and "result" in result_vars:
            new_body.append(ast.Return(value=ast.Name(id="result", ctx=ast.Load())))


        ast.fix_missing_locations(tree)
        tree.body = new_body
        ast.fix_missing_locations(tree)

        # ─── Wrap as async def __main() ──────────────────────────────
        func_def = ast.AsyncFunctionDef(
            name="__main",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=tree.body,
            decorator_list=[]
        )
        wrapper = ast.Module(body=[func_def], type_ignores=[])
        ast.fix_missing_locations(wrapper)

        if multi_mcp is None:
            # Hardened default: no live MCP handles are needed, so run the
            # transformed module in an isolated subprocess instead of exec'ing
            # it in-process (AST inspection alone is bypassable).
            return await _run_user_code_in_subprocess(
                ast.unparse(wrapper), session_id, start_time, start_timestamp
            )

        # In-process fallback: generated code may call live MCP tools by name
        # (tool_funcs proxies / parallel()), which only exist inside this
        # process. It still runs check_code_safety, but AST checks are
        # bypassable, so treat this path as more trusted than the subprocess.
        sandbox = build_safe_globals(tool_funcs, multi_mcp, session_id)
        local_vars = {}

        compiled = compile(wrapper, filename="<user_code>", mode="exec")
        exec(compiled, sandbox, local_vars)

        # ─── Execute and collect result ──────────────────────────────
        timeout = max(3, func_count * TIMEOUT_PER_FUNCTION)
        
        # Capture stdout/stderr
        log_capture = io.StringIO()
        
        with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
            returned = await asyncio.wait_for(local_vars["__main"](), timeout=timeout)

        result_value = {}

        if isinstance(returned, dict) and list(returned.keys()) == ["result"]:
            result_value = {"result": serialize_result_value(returned["result"])}
        if isinstance(returned, dict):
            result_value = {k: serialize_result_value(v) for k, v in returned.items()}

            # Check for MCP tool failures or error messages
            error_msg = _result_error_message(result_value, returned)
            if error_msg:
                return {
                    "status": "error",
                    "error": error_msg,
                    "execution_time": start_timestamp,
                    "total_time": str(round(time.perf_counter() - start_time, 3))
                }

        else:
            result_value = {"result": serialize_result_value(returned)}

        # import pdb; pdb.set_trace()

        

        log_json_block("Executor result", result_value)

        save_session_vars(session_id, result_value)
        # import pdb; pdb.set_trace()

        return {
            "status": "success",
            "result": result_value,
            "raw": result_value,
            "logs": log_capture.getvalue(),
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }

    except asyncio.TimeoutError:
        return {
            "status": "error",
            "error": f"Execution timed out after {func_count * TIMEOUT_PER_FUNCTION} seconds",
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }
    except Exception as e:
        print("⚠️ Code execution error:\n", traceback.format_exc())
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc(),
            "execution_time": start_timestamp,
            "total_time": str(round(time.perf_counter() - start_time, 3))
        }
