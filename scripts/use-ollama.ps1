param(
    [switch]$NoDocker,
    [switch]$LocalApi,
    [switch]$RestartWorker,
    [switch]$SkipPreflight
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$DockerEnvPath = Join-Path $Root ".env.docker.ollama"

if (-not (Test-Path $DockerEnvPath)) {
    throw "Missing Docker env file: $DockerEnvPath"
}

function Invoke-Compose {
    param(
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    & docker compose --env-file $DockerEnvPath @Args
}

Push-Location $Root
try {
    if (-not $NoDocker) {
        if ($LocalApi) {
            Invoke-Compose -Args @("--profile", "ollama", "up", "-d", "ollama")
            Write-Host "Local API mode selected; skipping Docker api/worker recreate."
        }
        else {
            Invoke-Compose -Args @("--profile", "ollama", "up", "-d", "ollama", "redis")
            Invoke-Compose -Args @("--profile", "ollama", "up", "-d", "--force-recreate", "api")

            $workerId = (& docker compose --env-file $DockerEnvPath ps -q worker)
            if ($RestartWorker -or $workerId) {
                Invoke-Compose -Args @("--profile", "ollama", "up", "-d", "--force-recreate", "worker")
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Ollama mode selected."
Write-Host "Compose env file: $DockerEnvPath"
Write-Host "Profile: local-laptop-gemma"
Write-Host "Ollama URL (Docker): http://ollama:11434"
Write-Host "Models: gemma4:e4b (local-laptop-gemma) or gemma3:4b (local-laptop-gemma-docker), plus nomic-embed-text"
Write-Host "For non-Docker local shell: `$env:S18_PROFILE='local-laptop-gemma'; `$env:OLLAMA_BASE_URL='http://127.0.0.1:11434'"

if (-not $SkipPreflight -and -not $LocalApi) {
    & (Join-Path $PSScriptRoot "preflight-runtime.ps1") -Mode ollama
}
