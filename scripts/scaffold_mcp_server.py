"""Scaffold a new MCP server template for S18.

Usage:
    python scripts/scaffold_mcp_server.py --name weather
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _normalize_server_name(raw: str) -> str:
    name = (raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name:
        raise ValueError("Server name cannot be empty.")
    if not name.replace("_", "").isalnum():
        raise ValueError("Server name must be alphanumeric/underscore only.")
    if name[0].isdigit():
        raise ValueError("Server name cannot start with a digit.")
    return name


def _render_server_py(server_name: str) -> str:
    server_title = server_name.replace("_", " ").title()
    return f'''"""MCP server scaffold for {server_title}."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("{server_name}")


@mcp.tool()
def ping() -> str:
    """Health check tool for scaffold validation."""
    return "{server_name} MCP server is running."


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def _render_readme(server_name: str) -> str:
    filename = f"server_{server_name}.py"
    return f"""# {server_name} MCP Server

Generated with `scripts/scaffold_mcp_server.py`.

## Run locally

```bash
python mcp_servers/custom/{server_name}/{filename}
```

## Register in S18

1. Add server config in `config/settings.json` (or dynamic API):
   - command: `python`
   - args: `["mcp_servers/custom/{server_name}/{filename}"]`
2. Add this server name to required agents in `config/agent_config.yaml`.
3. Refresh server metadata through `POST /mcp/refresh/{server_name}`.

## Next steps

- Replace `ping()` with domain tools.
- Add tool-level validation and guardrails.
- Add tests for your tool contracts.
"""


def scaffold(server_name: str, output_dir: Path, force: bool) -> Path:
    target_dir = output_dir / server_name
    if target_dir.exists() and not force:
        raise FileExistsError(
            f"Target already exists: {target_dir}. Use --force to overwrite files."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    files = {
        target_dir / "__init__.py": "",
        target_dir / f"server_{server_name}.py": _render_server_py(server_name),
        target_dir / "README.md": _render_readme(server_name),
        target_dir / "requirements.txt": "mcp[cli]>=1.6.0\n",
    }

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a new MCP server scaffold under mcp_servers/custom."
    )
    parser.add_argument("--name", required=True, help="Server name, e.g. weather")
    parser.add_argument(
        "--output-dir",
        default="mcp_servers/custom",
        help="Base output directory for scaffolded server.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if directory already exists.",
    )
    args = parser.parse_args()

    server_name = _normalize_server_name(args.name)
    target = scaffold(server_name, Path(args.output_dir), args.force)
    print(f"Scaffold created: {target}")


if __name__ == "__main__":
    main()
