# Phoenix Setup (Optional)

S18 includes an optional Arize Phoenix service in `monitoring/docker-compose.monitoring.yml`
for trace visualization.

## Start Phoenix with monitoring stack

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

Open UI at [http://localhost:6006](http://localhost:6006).

## Current trace posture

- S18 propagates integration metadata (`integration_id`, `workflow_id`, `contract_version`)
  across MCP call paths.
- Prometheus remains the primary metrics source for latency/error/success.
- Phoenix is available as the open-source trace dashboard endpoint in the stack.
