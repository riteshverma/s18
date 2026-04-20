# EAGV2 S18 Orchestration Rehearsal and Capture Guide

## Objective

Record one clean orchestration segment showing:

1. Run submission to `POST /runs`
2. Run polling via `GET /runs/{id}`
3. Completed run with structured output context

## Start Commands

From `S18Share/S18Share`:

```bash
uv sync
uv run python api.py
```

Alternative:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## Segment Capture Script (S18)

### Segment S1 (20-30s): Endpoint Context

- Open `http://localhost:8000/docs`.
- Briefly show `POST /runs` and `GET /runs/{id}`.
- Narrate: "This API is the orchestration layer that plans and executes multi-agent runs."

### Segment S2 (40-60s): Submit Fast Run

Use this body (also stored in wise-ai demo payload file):

```json
{
  "query": "[Patient ID: eagv2-demo-001] [Execution Mode: fast] Request: {\"hemoglobin\": 9.2, \"wbc\": 11200, \"rbc\": 3.8, \"platelets\": 410000}"
}
```

- Execute `POST /runs`.
- Capture returned run ID on screen.

Acceptance check:

- Response includes a non-empty `id`.

### Segment S3 (45-70s): Poll Run Lifecycle

Poll `GET /runs/{id}` until status reaches `completed` (or `failed` if you are recording fallback).

Acceptance check:

- You visibly show at least one non-terminal status and one terminal status.

### Segment S4 (20-30s): Explain Orchestration Value

- Highlight run response fields (status/graph/output summary as available).
- Narrate: "Planner and specialist agents execute through MCP-routed tools, then return structured output."

## PowerShell Polling Snippet (Copy/Paste)

```powershell
$base = "http://localhost:8000"
$body = @{
  query = "[Patient ID: eagv2-demo-001] [Execution Mode: fast] Request: {`"hemoglobin`": 9.2, `"wbc`": 11200, `"rbc`": 3.8, `"platelets`": 410000}"
} | ConvertTo-Json

$run = Invoke-RestMethod -Uri "$base/runs" -Method Post -ContentType "application/json" -Body $body
$id = $run.id
Write-Host "Run started: $id"

for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 3
  $statusResp = Invoke-RestMethod -Uri "$base/runs/$id" -Method Get
  Write-Host ("[{0}s] status={1}" -f (($i + 1) * 3), $statusResp.status)
  if ($statusResp.status -eq "completed" -or $statusResp.status -eq "failed") {
    $statusResp | ConvertTo-Json -Depth 8
    break
  }
}
```

## Clip Naming Convention

- `S1_runs_endpoint_context_take01.mp4`
- `S2_submit_run_take01.mp4`
- `S3_poll_lifecycle_take01.mp4`
- `S4_orchestration_explain_take01.mp4`

## Retake Triggers

- Wrong port confusion (`8000` local vs `8001` in Docker mapping)
- ID not visible in submitted run response
- Polling sequence does not show transition behavior
