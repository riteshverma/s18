param(
    [switch]$NoDocker,
    [switch]$HostServer,
    [switch]$RestartWorker,
    [switch]$SkipPreflight,
    [long]$MinimumModelBytes = 1GB
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$DockerEnvPath = if ($HostServer) {
    Join-Path $Root ".env.docker.llama-cpp-host"
}
else {
    Join-Path $Root ".env.docker.llama-cpp"
}

if (-not (Test-Path $DockerEnvPath)) {
    throw "Missing Docker env file: $DockerEnvPath"
}

if (-not $HostServer -and -not $NoDocker) {
    $modelFileName = if ($env:LLAMA_CPP_MODEL_FILE) { $env:LLAMA_CPP_MODEL_FILE } else { "model.gguf" }
    $modelPath = Join-Path (Join-Path $Root "models") $modelFileName
    if (-not (Test-Path $modelPath)) {
        throw "Missing llama.cpp model: $modelPath. Put a valid GGUF there, set LLAMA_CPP_MODEL_FILE, or use .\scripts\use-ollama.ps1."
    }

    $modelSize = (Get-Item $modelPath).Length
    if ($MinimumModelBytes -gt 0 -and $modelSize -lt $MinimumModelBytes) {
        $sizeMb = [math]::Round($modelSize / 1MB, 1)
        throw "llama.cpp model '$modelFileName' is only ${sizeMb} MB. A 7B GGUF is usually several GB; this file is likely incomplete/corrupt. Re-download it, set LLAMA_CPP_MODEL_FILE to a valid GGUF, pass -MinimumModelBytes 0 for tiny models, or use .\scripts\use-ollama.ps1."
    }
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
        if ($HostServer) {
            Invoke-Compose -Args @("up", "-d", "redis")
            Invoke-Compose -Args @("up", "-d", "--force-recreate", "api")
        }
        else {
            Invoke-Compose -Args @("--profile", "llama_cpp", "up", "-d", "llama_cpp", "redis")
            Invoke-Compose -Args @("--profile", "llama_cpp", "up", "-d", "--force-recreate", "api")
        }

        $workerId = (& docker compose --env-file $DockerEnvPath ps -q worker)
        if ($RestartWorker -or $workerId) {
            if ($HostServer) {
                Invoke-Compose -Args @("up", "-d", "--force-recreate", "worker")
            }
            else {
                Invoke-Compose -Args @("--profile", "llama_cpp", "up", "-d", "--force-recreate", "worker")
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host "llama.cpp mode selected."
Write-Host "Profile: local-llama-cpp"
Write-Host "Compose env file: $DockerEnvPath"
if ($HostServer) {
    Write-Host "llama.cpp URL for Docker: http://host.docker.internal:8080"
}
else {
    Write-Host "llama.cpp URL for Docker: http://s18share-llama-cpp:8080"
}
Write-Host "For a non-Docker local shell, run: `$env:S18_PROFILE='local-llama-cpp'; `$env:LLAMA_CPP_BASE_URL='http://127.0.0.1:8080'"

if (-not $SkipPreflight) {
    & (Join-Path $PSScriptRoot "preflight-runtime.ps1") -Mode llama_cpp
}
