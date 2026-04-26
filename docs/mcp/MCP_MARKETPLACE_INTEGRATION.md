# MCP Marketplace Integration

S18 can act as a primary MCP orchestration hub by connecting built-in servers
and external servers from the Model Context Protocol ecosystem.

Reference server catalog: [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)

## Why this matters

- Keep one orchestration core while swapping tool backends.
- Add domain capabilities without changing agent loop internals.
- Trace tool usage by integration/workflow metadata.

## Plug in an external MCP server

1. Add the server configuration via API:

```bash
curl -X POST "http://localhost:8000/mcp/servers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_external_server",
    "config": {
      "command": "python",
      "args": ["path/to/server.py"],
      "transport": "stdio"
    }
  }'
```

2. Confirm server registration:

```bash
curl "http://localhost:8000/mcp/servers"
```

3. Refresh tool metadata:

```bash
curl -X POST "http://localhost:8000/mcp/refresh/my_external_server"
```

## Recommended contract for external servers

- Expose concise tool names and clear argument schemas.
- Return structured JSON where possible.
- Keep tools bounded in runtime and side effects.
- Include README docs near the server implementation.

## One-click server scaffold

Generate a starter server with:

```bash
python scripts/scaffold_mcp_server.py --name weather
```

The command creates `mcp_servers/custom/weather/` with:
- runnable `server_weather.py`
- server README
- minimal requirements file
