param(
    [string]$Container = "s18share-api",
    [switch]$RestartOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ServerRag = Join-Path $Root "mcp_servers\server_rag.py"
$RouterRag = Join-Path $Root "routers\rag.py"

function Wait-ApiHealth {
    param(
        [int]$MaxAttempts = 45,
        [int]$DelaySeconds = 2
    )

    for ($i = 0; $i -lt $MaxAttempts; $i++) {
        $health = & curl.exe -sS -m 5 "http://localhost:8001/health" 2>$null
        if ($LASTEXITCODE -eq 0 -and $health) {
            return $health
        }
        Start-Sleep -Seconds $DelaySeconds
    }

    throw "API health check timed out after $($MaxAttempts * $DelaySeconds)s."
}

if (-not $RestartOnly) {
    if (-not (Test-Path $ServerRag)) { throw "Missing file: $ServerRag" }
    if (-not (Test-Path $RouterRag)) { throw "Missing file: $RouterRag" }

    docker cp "$ServerRag" "${Container}:/app/mcp_servers/server_rag.py"
    if ($LASTEXITCODE -ne 0) { throw "docker cp failed for server_rag.py" }

    docker cp "$RouterRag" "${Container}:/app/routers/rag.py"
    if ($LASTEXITCODE -ne 0) { throw "docker cp failed for routers/rag.py" }
}

docker restart $Container | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Failed to restart container '$Container'" }

$healthJson = Wait-ApiHealth
$health = $healthJson | ConvertFrom-Json

docker exec $Container python -c "s=open('/app/mcp_servers/server_rag.py',encoding='utf-8').read(); print('guard', 'INDEX_MANIFEST_NAME' in s, 'status_field', 'index_compatible' in s)"
if ($LASTEXITCODE -ne 0) { throw "Guard verification probe failed in container '$Container'." }

Write-Host "RAG sync complete. API status=$($health.status), mcp_ready=$($health.mcp_ready)."
