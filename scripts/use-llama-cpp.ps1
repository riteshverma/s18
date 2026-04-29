param(
    [switch]$NoDocker,
    [switch]$HostServer,
    [switch]$RestartWorker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$EnvPath = Join-Path $Root ".env"

function Set-DotEnvValue {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )

    $line = "$Key=$Value"
    $escapedKey = [regex]::Escape($Key)
    $lines = @()
    if (Test-Path $EnvPath) {
        $lines = @(Get-Content -Path $EnvPath)
    }

    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$escapedKey\s*=") {
            $lines[$i] = $line
            $found = $true
        }
    }

    if (-not $found) {
        $lines += $line
    }

    Set-Content -Path $EnvPath -Value $lines -Encoding UTF8
}

Set-DotEnvValue -Key "S18_PROFILE" -Value "local-llama-cpp"
Set-DotEnvValue -Key "OLLAMA_BASE_URL" -Value "http://ollama:11434"

$LlamaCppUrl = "http://llama_cpp:8080"
if ($HostServer) {
    $LlamaCppUrl = "http://host.docker.internal:8080"
}
Set-DotEnvValue -Key "LLAMA_CPP_BASE_URL" -Value $LlamaCppUrl

Push-Location $Root
try {
    if (-not $NoDocker) {
        if ($HostServer) {
            docker compose up -d redis
            docker compose up -d --force-recreate api
        }
        else {
            docker compose --profile llama_cpp up -d llama_cpp redis
            docker compose --profile llama_cpp up -d --force-recreate api
        }

        $workerId = docker compose ps -q worker
        if ($RestartWorker -or $workerId) {
            if ($HostServer) {
                docker compose up -d --force-recreate worker
            }
            else {
                docker compose --profile llama_cpp up -d --force-recreate worker
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "llama.cpp mode selected."
Write-Host "Profile: local-llama-cpp"
Write-Host "llama.cpp URL for Docker: $LlamaCppUrl"
Write-Host "For a non-Docker local shell, run: `$env:S18_PROFILE='local-llama-cpp'; `$env:LLAMA_CPP_BASE_URL='http://127.0.0.1:8080'"
