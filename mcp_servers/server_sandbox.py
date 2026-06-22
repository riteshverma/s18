import asyncio
import sys
import os
from pathlib import Path

# Fix Path: Add 'Arcturus' to sys.path so we can import 'tools' and 'core'
# Current file: .../Arcturus/mcp_servers/server_sandbox.py
# We want: .../Arcturus/
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# DEBUG
sys.stderr.write(f"DEBUG: Added to path: {root_dir}\n")
sys.stderr.write(f"DEBUG: Contents of root: {os.listdir(root_dir)}\n")
import importlib.util
try:
    from stdio_safety import configure_mcp_stdio_logging
except ImportError:
    from .stdio_safety import configure_mcp_stdio_logging

# Use importlib to avoid conflict with 'mcp_servers.tools'
sandbox_path = root_dir / "tools/sandbox.py"
spec = importlib.util.spec_from_file_location("tools.sandbox", sandbox_path)
sandbox_mod = importlib.util.module_from_spec(spec)
# We need to register it so internal imports (if any) work?
# sandbox.py imports core.utils. 
# We already added root_dir to sys.path so core.utils should work.
spec.loader.exec_module(sandbox_mod)

run_user_code = sandbox_mod.run_user_code

workspace_io_path = root_dir / "tools" / "workspace_io.py"
workspace_spec = importlib.util.spec_from_file_location("tools.workspace_io", workspace_io_path)
workspace_io_mod = importlib.util.module_from_spec(workspace_spec)
workspace_spec.loader.exec_module(workspace_io_mod)

read_workspace_file_fn = workspace_io_mod.read_workspace_file
write_workspace_file_fn = workspace_io_mod.write_workspace_file

from mcp.server.fastmcp import FastMCP

configure_mcp_stdio_logging()

# Initialize FastMCP server
mcp = FastMCP("sandbox")

@mcp.tool()
async def run_python_script(code: str) -> str:
    """
    Execute Python code in a secure sandbox.
    Use this for math, data processing, and logic.
    Returns the stdout and result of the execution.
    """
    # We pass multi_mcp=None for now, limiting the sandbox to pure Python logic
    # without ability to call other MCP tools recursively.
    result = await run_user_code(code, multi_mcp=None, session_id="mcp_worker")
    
    # Format the output for the agent
    if result.get("status") == "success":
        # Return the 'result' key or raw stdout captured
        val = result.get("result", "")
        return f"Execution Successful:\n{val}"
    else:
        err = result.get("error", "Unknown error")
        return f"Execution Failed:\n{err}"


@mcp.tool()
def read_workspace_file(path: str, workspace_root: str = "") -> str:
    """
    Read a UTF-8 text file from the ClawBench task workspace.
    Paths may be relative to the workspace root or absolute within it.
    """
    try:
        root = workspace_root.strip() or None
        return read_workspace_file_fn(path, root)
    except Exception as exc:
        return f"Read failed: {exc}"


@mcp.tool()
def write_workspace_file(path: str, content: str, workspace_root: str = "") -> str:
    """
    Write UTF-8 text to a file inside the ClawBench task workspace.
    Creates parent directories as needed. Use for notes, patches, and deliverables.
    """
    try:
        root = workspace_root.strip() or None
        return write_workspace_file_fn(path, content, root)
    except Exception as exc:
        return f"Write failed: {exc}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
