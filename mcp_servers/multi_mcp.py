
import asyncio
import sys
import shutil
import json
import os
import subprocess
import builtins
import contextvars
from pathlib import Path

# Windows: ProactorEventLoop required for asyncio subprocess (uv run MCP server)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool
from rich import print as rich_print
from core.prometheus_metrics import MCP_TOOL_CALLS_TOTAL, MCP_TOOL_LATENCY_MS, elapsed_ms, now_ms
from config.settings_loader import (
    get_mcp_mode,
    get_mcp_required_servers,
    get_mcp_stdio_connect_timeout,
    settings,
)


_WORKSPACE_FILE_TOOLS = {"read_workspace_file", "write_workspace_file"}

class MultiMCP:
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}  # server_name -> session
        self.tools = {}     # server_name -> [Tool]
        self.start_results = {}
        self.start_completed = False
        
        # Robust path resolution
        self.base_dir = Path(__file__).parent
        self.config_path = self.base_dir / "mcp_config.json"
        
        # Metadata Cache (for tools)
        self.cache_path = self.base_dir.parent / "config" / "mcp_cache.json"
        self._cached_metadata = self._load_cache()
        
        self.server_configs = self._load_config()
        
        # Disabled tools cache
        self.disabled_tools = set() # { "server:tool" }
        self.disabled_tools_path = self.base_dir.parent / "config" / "disabled_tools.json"
        self._load_disabled_tools()
        self._trace_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
            "mcp_trace_context",
            default={},
        )

    def get_mode(self) -> str:
        """Return MCP operating mode."""
        return get_mcp_mode()

    def is_strict_mode(self) -> bool:
        """Return whether strict MCP mode is enabled."""
        return self.get_mode() == "strict"

    def get_required_servers(self) -> list[str]:
        """Return MCP servers required for strict mode readiness."""
        return get_mcp_required_servers()

    def should_use_cached_metadata(self) -> bool:
        """Legacy mode prefers cache to preserve current startup behavior."""
        return not self.is_strict_mode()

    def _set_server_result(self, name: str, status: str, detail: str | None = None):
        """Record MCP server startup result."""
        result = {"status": status}
        if detail:
            result["detail"] = detail
        self.start_results[name] = result

    def get_health_status(self) -> dict:
        """Return MCP readiness details for health endpoints."""
        connected_servers = sorted(self.get_connected_servers())
        required_servers = self.get_required_servers()
        required_connected = all(
            server in self.sessions for server in required_servers
        )
        if self.is_strict_mode():
            mcp_ready = self.start_completed and required_connected
        else:
            mcp_ready = True
        return {
            "mode": self.get_mode(),
            "mcp_ready": mcp_ready,
            "start_completed": self.start_completed,
            "connected_servers": connected_servers,
            "required_servers": required_servers,
            "start_results": dict(self.start_results),
        }

    def set_trace_context(self, trace_context: dict | None):
        """Set request-scoped metadata and return reset token."""
        return self._trace_context.set(trace_context or {})

    def reset_trace_context(self, token):
        """Reset request-scoped metadata with the provided token."""
        if token is not None:
            self._trace_context.reset(token)

    def _load_config(self) -> dict:
        """Load server configuration from JSON"""
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except Exception as e:
                print(f"⚠️ Failed to load MCP config: {e}")
        return {}

    def _save_config(self):
        """Save current server configuration"""
        try:
            self.config_path.write_text(json.dumps(self.server_configs, indent=2))
        except Exception as e:
            print(f"⚠️ Failed to save MCP config: {e}")

    def _load_disabled_tools(self):
        if self.disabled_tools_path.exists():
            try:
                data = json.loads(self.disabled_tools_path.read_text())
                self.disabled_tools = set(data)
            except: pass

    def _save_disabled_tools(self):
        self.disabled_tools_path.write_text(json.dumps(list(self.disabled_tools)))
        
    def set_tool_state(self, server_name: str, tool_name: str, enabled: bool):
        key = f"{server_name}:{tool_name}"
        if enabled:
            if key in self.disabled_tools:
                self.disabled_tools.remove(key)
                self._save_disabled_tools()
        else:
            self.disabled_tools.add(key)
            self._save_disabled_tools()

    async def add_server(self, name: str, config: dict):
        """Dynamically add a new server"""
        if name in self.sessions:
            raise ValueError(f"Server '{name}' already exists")
        
        self.server_configs[name] = config
        self._save_config()
        
        # Start immediately
        await self._start_server(name, config)
        return True

    async def remove_server(self, name: str):
        """Remove a server"""
        try:
            if name in self.server_configs:
                del self.server_configs[name]
                self._save_config()
            
            # Remove from active sessions/tools regardless of config presence
            if name in self.sessions:
                # We can't strictly 'close' the session easily without closing the whole stack
                # unless we manage per-session exit stacks (which would be better but complex refactor)
                # For now, just removing it prevents further routing.
                print(f"  🗑️ Removed server '{name}' from sessions")
                del self.sessions[name]
                
            if name in self.tools:
                del self.tools[name]
                
            return True
        except Exception as e:
            print(f"  ⚠️ Error removing server {name}: {e}")
            # Still return True if we managed to at least remove it from config? 
            # Or False? Let's return True effectively as "we tried our best to forget it"
            return True

    async def _start_server(self, name: str, config: dict):
        """Start a single server with timeout protection"""
        # Skip if explicitly disabled
        if config.get("enabled", True) is False:
            print(f"  ⏭️ [dim]Server '{name}' is disabled in config. Skipping.[/dim]")
            self._set_server_result(name, "disabled")
            return False

        try:
            cmd = config.get("command", "uv")
            args = config.get("args", [])
            server_type = config.get("type", "local-script")
            env = config.get("env", None) # Optional env vars

            # --- Pre-processing for different types ---
            
            if server_type == "local-script":
                # Ensure we point to the script in this directory
                script_name = args[-1] # Assume last arg is script
                if not Path(script_name).is_absolute() and (self.base_dir / script_name).exists():
                     # Reconstruct args with absolute path
                     # args usually: ["run", "server_browser.py"]
                     script_path = str(self.base_dir / script_name)
                     args = args[:-1] + [script_path]

            elif server_type == "stdio-git":
                # Clone repo and setup
                repo_url = config.get("source")
                if not repo_url:
                    raise ValueError("Missing 'source' (git url) for stdio-git server")
                
                server_dir = self.base_dir.parent / "data" / "mcp_repos" / name
                server_dir.parent.mkdir(parents=True, exist_ok=True)
                
                if not server_dir.exists():
                     print(f"  ⬇️ Cloning {name} from {repo_url}...")
                     # Use sync subprocess in a thread to avoid Windows asyncio subprocess issues
                     def _git_clone():
                         r = subprocess.run(
                             ["git", "clone", repo_url, str(server_dir)],
                             capture_output=True, text=True, timeout=120
                         )
                         if r.returncode != 0:
                             raise RuntimeError(f"Git clone failed for {name}: {r.stderr or r.stdout}")
                     await asyncio.to_thread(_git_clone)


                # Configure command to run from that directory with uv
                # We typically run `uv run --directory <repo> <script>`
                cmd = "uv"
                
                if cmd == "uv" and "run" in args:
                     # Inject --directory <path> after 'run' 
                     # args is likely ["run", "script.py"]
                     # We want ["run", "--directory", str(server_dir), "script.py"]
                     
                     # Find index of 'run'
                     try:
                         run_idx = args.index("run")
                         # Insert directory args after run
                         args.insert(run_idx + 1, "--directory")
                         args.insert(run_idx + 2, str(server_dir))
                         
                         # Check for requirements.txt to install dependencies automatically
                         req_file = server_dir / "requirements.txt"
                         if req_file.exists():
                             args.insert(run_idx + 3, "--with-requirements")
                             args.insert(run_idx + 4, str(req_file))
                             print(f"  📦 Detected requirements.txt for {name}, auto-installing dependencies...")
                         
                         # --- Smart Entry Point Detection ---
                         # The config might default to 'src/server.py', but the repo might use 'yfinance_mcp_server.py'
                         # We check the LAST argument which is usually the script path
                         script_arg_idx = -1
                         current_script = args[script_arg_idx]
                         
                         # Construct full path to check
                         script_path = server_dir / current_script
                         if not script_path.exists():
                             print(f"  ⚠️ Configured script '{current_script}' not found in {name}. Attempting auto-detection...")
                             
                             # Search candidates
                             candidates = list(server_dir.glob("*_mcp_server.py")) + \
                                          list(server_dir.glob("server.py")) + \
                                          list(server_dir.glob("src/server.py")) + \
                                          list(server_dir.glob("*.py"))
                             
                             # Filter out non-server looking files if possible, but taking the first specific match is good
                             best_candidate = None
                             for c in candidates:
                                 # Prefer *mcp_server.py or server.py
                                 if "mcp_server" in c.name or c.name == "server.py":
                                     best_candidate = c
                                     break
                             
                             if not best_candidate and candidates:
                                 # Fallback to first python file if it looks like a server?
                                 # Just take the first one (often there's only one main script in simple repos)
                                 best_candidate = candidates[0]
                             
                             if best_candidate:
                                 # Update args
                                 new_script = str(best_candidate.relative_to(server_dir))
                                 args[script_arg_idx] = new_script
                                 print(f"  ✅ Auto-detected entry point: {new_script}")
                             else:
                                 print(f"  ❌ Could not auto-detect entry point for {name}")

                     except ValueError:
                         pass
            
            # --- Execution ---

            final_env = os.environ.copy()
            if env:
                final_env.update(env)

            # Optional working directory (e.g. GBrain needs repo root for TS imports)
            cwd_param = None
            cwd_cfg = config.get("cwd")
            if cwd_cfg:
                cwd_path = Path(cwd_cfg)
                if not cwd_path.is_absolute():
                    cwd_path = self.base_dir.parent / cwd_path
                cwd_param = str(cwd_path.resolve())
                if not Path(cwd_param).exists():
                    detail = f"cwd does not exist: {cwd_param}"
                    builtins.print(f"[MCP] {name} skipped: {detail}")
                    self._set_server_result(name, "skipped", detail)
                    return False

            # Resolve Bun when not on PATH (typical on fresh Windows installs)
            if cmd == "bun" and not shutil.which("bun"):
                bun_exe = Path.home() / ".bun" / "bin" / ("bun.exe" if sys.platform == "win32" else "bun")
                if bun_exe.is_file():
                    cmd = str(bun_exe)
                else:
                    detail = "'bun' not found on PATH."
                    builtins.print(f"[MCP] {name} skipped: {detail}")
                    self._set_server_result(name, "skipped", detail)
                    return False

            # Check if uv exists fallback
            if cmd == "uv" and not shutil.which("uv"):
                cmd = sys.executable
                # This fallback is flaky for complex args, keep simple
                if args[0] == "run":
                     # If falling back to python, we need to handle the directory/cwd manually
                     # or just hope it works?
                     # Ideally we shouldn't fallback for git repos if they rely on uv dependencies
                     print(f"  ⚠️ 'uv' not found. Falling back to system python is risky for {name}.")
                     # Try to fix path to be absolute if we are not using uv (and not changing cwd)
                     # But we can't easily change cwd for just this process with StdioServerParameters efficiently?
                     # Actually we can just run python <full_path_to_script>
                     # Remove 'run', '--directory', etc.
                     # This is Getting Complicated. Let's assume UV exists for 'stdio-git'.
                     pass # Rely on uv being present

            # Skip gracefully if command is unavailable in current runtime.
            cmd_path = Path(cmd)
            if not cmd_path.is_absolute() and not shutil.which(cmd):
                detail = f"command not found: {cmd}"
                builtins.print(f"[MCP] {name} skipped: {detail}")
                self._set_server_result(name, "skipped", detail)
                return False


            server_params = StdioServerParameters(
                command=cmd,
                args=args,
                env=final_env,
                cwd=cwd_param,
            )
            
            # Connect with timeout (spawn + MCP initialize); cold `uv run` can exceed a few seconds.
            cfg_timeout = config.get("stdio_connect_timeout_seconds")
            try:
                connect_timeout = (
                    float(cfg_timeout)
                    if cfg_timeout is not None
                    else float(get_mcp_stdio_connect_timeout())
                )
            except (TypeError, ValueError):
                connect_timeout = float(get_mcp_stdio_connect_timeout())
            if connect_timeout <= 0:
                connect_timeout = float(get_mcp_stdio_connect_timeout())

            async with asyncio.timeout(connect_timeout):
                read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
                session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                # List tools
                if self.should_use_cached_metadata() and name in self._cached_metadata:
                    # Use ASCII-only logging via standard print to avoid Windows console encoding issues
                    builtins.print(f"[MCP] {name} tools loaded from cache.")
                    cached_tools = []
                    for t_dict in self._cached_metadata[name]:
                        cached_tools.append(Tool(
                            name=t_dict["name"],
                            description=t_dict["description"],
                            inputSchema=t_dict["inputSchema"]
                        ))
                    self.tools[name] = cached_tools
                    self._set_server_result(name, "connected", "metadata_source=cache")
                else:
                    result = await session.list_tools()
                    self.tools[name] = result.tools
                    self._save_to_cache(name, result.tools)
                    builtins.print(f"[MCP] {name} connected. Tools: {len(result.tools)}")
                    self._set_server_result(name, "connected", "metadata_source=live")
                
                self.sessions[name] = session

        except TimeoutError:
             builtins.print(f"[MCP] {name} timed out during startup.")
             self._set_server_result(name, "timeout")
        except Exception as e:
            import traceback
            builtins.print(f"[MCP] {name} failed to start: {e}")
            self._set_server_result(name, "failed", str(e))
            traceback.print_exc()
        except BaseException as e:
            builtins.print(f"[MCP] {name} CRITICAL FAILURE: {e}")
            self._set_server_result(name, "failed", str(e))

    async def start(self):
        """Start all configured servers"""
        # Use plain ASCII to avoid Windows console encoding issues with emojis
        builtins.print("[MCP] Starting MCP Servers...")
        self.start_results = {}
        self.start_completed = False
        for name, config in self.server_configs.items():
            if config.get("enabled", True):
                await self._start_server(name, config)
            else:
                builtins.print(f"[MCP] Skipping disabled server: {name}")
                self._set_server_result(name, "disabled")
        self.start_completed = True
        if self.is_strict_mode():
            missing_required = [
                server for server in self.get_required_servers()
                if server not in self.sessions
            ]
            if missing_required:
                raise RuntimeError(
                    "Strict MCP mode requires connected servers: "
                    + ", ".join(sorted(missing_required))
                )

    async def stop(self):
        """Stop all servers"""
        builtins.print("[MCP] Stopping MCP Servers...")
        await self.exit_stack.aclose()

    def get_all_tools(self) -> list:
        """Get all tools from all connected servers"""
        all_tools = []
        for tools in self.tools.values():
            all_tools.extend(tools)
        return all_tools
    
    def get_connected_servers(self) -> list:
        """Return list of connected server names"""
        return list(self.sessions.keys())

    async def function_wrapper(self, tool_name: str, *args):
        """Execute a tool using positional arguments by mapping them to schema keys"""
        # Find tool definition
        target_tool = None
        for tools in self.tools.values():
            for tool in tools:
                if tool.name == tool_name:
                    target_tool = tool
                    break
            if target_tool: break
        
        if not target_tool:
            return f"Error: Tool {tool_name} not found"

        # Map positional args to keyword args based on schema
        arguments = {}
        schema = target_tool.inputSchema
        if schema and 'properties' in schema:
            keys = list(schema['properties'].keys())
            for i, arg in enumerate(args):
                if i < len(keys):
                    arguments[keys[i]] = arg
        
        try:
            result = await self.route_tool_call(tool_name, arguments)
            # Unpack CallToolResult
            if hasattr(result, 'content') and result.content:
                return result.content[0].text
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def get_tools_from_servers(self, server_names: list) -> list:
        """Get flattened list of tools from requested servers"""
        all_tools = []
        for name in server_names:
            if name in self.tools:
                # Filter out disabled tools
                for tool in self.tools[name]:
                    key = f"{name}:{tool.name}"
                    if key not in self.disabled_tools:
                        all_tools.append(tool)
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict):
        """Call a tool on a specific server"""
        if server_name not in self.sessions:
            raise ValueError(f"Server '{server_name}' not connected")
        trace_context = self._trace_context.get()
        arguments = self._prepare_tool_arguments(tool_name, arguments, trace_context)
        if trace_context:
            print(
                f"[MCP trace] server={server_name} tool={tool_name} "
                f"integration_id={trace_context.get('integration_id', 'default')} "
                f"workflow_id={trace_context.get('workflow_id', 'generic')} "
                f"contract_version={trace_context.get('contract_version', 'v1')}"
            )
        return await self.sessions[server_name].call_tool(tool_name, arguments)

    def _prepare_tool_arguments(
        self,
        tool_name: str,
        arguments: dict | None,
        trace_context: dict | None = None,
    ) -> dict:
        """Bind sensitive tool arguments to trusted request-scoped context."""
        prepared = dict(arguments or {})
        if tool_name not in _WORKSPACE_FILE_TOOLS:
            return prepared

        context = trace_context if trace_context is not None else self._trace_context.get()
        trusted_workspace = (context or {}).get("workspace")
        if not trusted_workspace:
            prepared.pop("workspace_root", None)
            raise ValueError(
                f"Tool '{tool_name}' requires a trusted workspace context"
            )

        prepared["workspace_root"] = trusted_workspace
        return prepared

    # Helper to route tool call by finding which server has it
    async def route_tool_call(self, tool_name: str, arguments: dict):
        from core.circuit_breaker import get_breaker, CircuitOpenError
        start_ms = now_ms()
        trace_context = self._trace_context.get()
        integration_id = trace_context.get("integration_id", "default")
        workflow_id = trace_context.get("workflow_id", "generic")
        contract_version = trace_context.get("contract_version", "v1")
        
        # Get or create circuit breaker for this tool
        breaker = get_breaker(tool_name, failure_threshold=5, recovery_timeout=60.0)
        
        # Check if circuit allows execution
        if not breaker.can_execute():
            status = breaker.get_status()
            raise CircuitOpenError(
                f"Circuit open for '{tool_name}' - service failing. "
                f"Retry in {status['time_until_retry']:.0f}s"
            )
        
        try:
            matching_servers = []
            for name, tools in self.tools.items():
                for tool in tools:
                    if tool.name != tool_name:
                        continue
                    key = f"{name}:{tool_name}"
                    if key in self.disabled_tools:
                        continue
                    matching_servers.append(name)

            if not matching_servers:
                raise ValueError(f"Tool '{tool_name}' not found in any enabled server")

            # Deterministic conflict resolution for EHR retrieval tools.
            if tool_name in {"get_patient_records", "search_labs"} and "mockehr" in matching_servers:
                selected_server = "mockehr"
            else:
                selected_server = sorted(matching_servers)[0]
                if len(matching_servers) > 1:
                    print(
                        f"  ⚠️ Tool collision for '{tool_name}', "
                        f"choosing '{selected_server}' from {matching_servers}"
                    )

            timeout_seconds = float(settings.get("mcp", {}).get("tool_timeout_seconds", 45))
            try:
                result = await asyncio.wait_for(
                    self.call_tool(selected_server, tool_name, arguments),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"MCP tool timeout tool={tool_name} server={selected_server} timeout_seconds={timeout_seconds}"
                ) from exc
            breaker.record_success()
            MCP_TOOL_CALLS_TOTAL.labels(
                tool=tool_name,
                status="success",
                integration_id=integration_id,
                workflow_id=workflow_id,
                contract_version=contract_version,
            ).inc()
            return result
        except CircuitOpenError:
            MCP_TOOL_CALLS_TOTAL.labels(
                tool=tool_name,
                status="circuit_open",
                integration_id=integration_id,
                workflow_id=workflow_id,
                contract_version=contract_version,
            ).inc()
            raise  # Re-raise circuit errors without recording failure
        except Exception as e:
            breaker.record_failure()
            MCP_TOOL_CALLS_TOTAL.labels(
                tool=tool_name,
                status="error",
                integration_id=integration_id,
                workflow_id=workflow_id,
                contract_version=contract_version,
            ).inc()
            raise
        finally:
            MCP_TOOL_LATENCY_MS.labels(
                tool=tool_name,
                integration_id=integration_id,
                workflow_id=workflow_id,
                contract_version=contract_version,
            ).observe(elapsed_ms(start_ms))

    def _load_cache(self) -> dict:
        """Load metadata cache from file"""
        if self.cache_path.exists():
            try:
                import json
                return json.loads(self.cache_path.read_text())
            except Exception as e:
                builtins.print(f"[MCP] Failed to load MCP cache: {e}")
        return {}

    def _save_to_cache(self, server_name: str, tools: list):
        """Save tool metadata to persistent cache"""
        try:
            import json
            # Ensure directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing
            cache = self._load_cache()
            
            # Update
            tool_list = []
            for t in tools:
                tool_list.append({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.inputSchema
                })
            cache[server_name] = tool_list
            
            # Write back
            self.cache_path.write_text(json.dumps(cache, indent=2))
            builtins.print(f"[MCP] Cached metadata for {server_name}")
        except Exception as e:
            builtins.print(f"[MCP] Failed to save MCP cache for {server_name}: {e}")

    async def refresh_server(self, server_name: str):
        """Force refresh tool metadata for a server"""
        if server_name in self.sessions:
            builtins.print(f"[MCP] Refreshing tools for {server_name}...")
            result = await self.sessions[server_name].list_tools()
            self.tools[server_name] = result.tools
            self._save_to_cache(server_name, result.tools)
            return True
        return False
        
    def get_server_readme(self, server_name: str) -> str:
        """Get the README content for a server"""
        config = self.server_configs.get(server_name)
        if not config:
            return None
            
        repo_path = None
        
        # Determine path based on type
        if config.get("type") == "stdio-git":
             repo_path = self.base_dir.parent / "data" / "mcp_repos" / server_name
        elif config.get("type") == "local-script":
             # Use the base dir
             repo_path = self.base_dir
        
        if repo_path:
            # Try potential readme names, prioritizing server-specific ones
            candidates = [
                f"README_{server_name}.md",
                f"docs/README_{server_name}.md",
                "README.md", 
                "readme.md", 
                "README.txt", 
                "README"
            ]
            
            for name in candidates:
                p = repo_path / name
                if p.exists():
                    return p.read_text(encoding="utf-8", errors="replace")
        
        return None

