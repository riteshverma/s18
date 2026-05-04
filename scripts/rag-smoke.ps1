param(
    [string]$Path = "docs/architecture/WISE_AI_CDSS_Architecture_2026-03.md",
    [string]$Query = "Summarize the main components of the WISE AI CDSS architecture.",
    [switch]$Force,
    [switch]$SyncFromDocs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Get-JsonFromCurl {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [string]$Method = "GET",
        [int]$TimeoutSeconds = 15
    )
    $args = @("-sS", "-m", "$TimeoutSeconds")
    if ($Method -eq "POST") { $args += @("-X", "POST") }
    $args += $Url
    $raw = & curl.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Url"
    }
    return $raw | ConvertFrom-Json
}

if ($SyncFromDocs) {
    $source = Join-Path $Root $Path
    $target = Join-Path (Join-Path $Root "data") $Path
    if (-not (Test-Path $source)) {
        throw "SyncFromDocs requested but source file not found: $source"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -Force $source $target
    Write-Host "Copied $source -> $target"
}

$health = Get-JsonFromCurl -Url "http://localhost:8001/health"
if ($health.status -ne "ok") {
    throw "API health is not ok."
}

$forceValue = if ($Force) { "true" } else { "false" }
$escapedPath = [uri]::EscapeDataString($Path)
$reindexUrl = "http://localhost:8001/rag/reindex?path=$escapedPath&force=$forceValue"
$reindex = Get-JsonFromCurl -Url $reindexUrl -Method "POST" -TimeoutSeconds 60
if ($reindex.status -ne "success") {
    throw "Reindex request did not succeed."
}
Write-Host "Reindex requested for '$Path' (force=$forceValue)."

$status = $null
for ($i = 0; $i -lt 60; $i++) {
    $status = Get-JsonFromCurl -Url "http://localhost:8001/rag/indexing_status"
    if (-not $status.active) { break }
    Start-Sleep -Seconds 2
}

if ($status.active) {
    throw "Indexing still active after wait timeout."
}
if ($status.last_error) {
    throw "Indexing completed with last_error: $($status.last_error)"
}

$searchUrl = "http://localhost:8001/rag/search?query=$([uri]::EscapeDataString($Query))"
$search = Get-JsonFromCurl -Url $searchUrl -TimeoutSeconds 45
if ($search.status -ne "success") {
    throw "Search status is not success."
}
if (-not $search.results -or $search.results.Count -lt 1) {
    throw "Search returned no results."
}

$first = $search.results[0]
Write-Host "Smoke OK: results=$($search.results.Count), first_source=$($first.source), page=$($first.page)"
