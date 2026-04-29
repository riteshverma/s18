param(
    [switch]$NoDocker,
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

Set-DotEnvValue -Key "S18_PROFILE" -Value "local-laptop-gemma"
Set-DotEnvValue -Key "OLLAMA_BASE_URL" -Value "http://ollama:11434"
Set-DotEnvValue -Key "LLAMA_CPP_BASE_URL" -Value "http://llama_cpp:8080"

Push-Location $Root
try {
    if (-not $NoDocker) {
        docker compose --profile ollama up -d ollama redis
        docker compose --profile ollama up -d --force-recreate api

        $workerId = docker compose ps -q worker
        if ($RestartWorker -or $workerId) {
            docker compose --profile ollama up -d --force-recreate worker
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Ollama mode selected."
Write-Host "Profile: local-laptop-gemma"
Write-Host "Ollama URL for Docker: http://ollama:11434"
Write-Host "For a non-Docker local shell, run: `$env:S18_PROFILE='local-laptop-gemma'"
