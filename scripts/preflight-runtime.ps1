param(
    [ValidateSet("ollama", "llama_cpp")]
    [string]$Mode = "ollama"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-ContainerRunning {
    param([Parameter(Mandatory = $true)][string]$Name)
    $state = docker inspect -f "{{.State.Running}}" $Name 2>$null
    return $LASTEXITCODE -eq 0 -and $state.Trim().ToLower() -eq "true"
}

function Invoke-InContainerPython {
    param(
        [Parameter(Mandatory = $true)][string]$Container,
        [Parameter(Mandatory = $true)][string]$Code
    )
    docker exec $Container python -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw "Python probe failed in container '$Container'."
    }
}

if (-not (Test-ContainerRunning "s18share-api")) {
    throw "s18share-api is not running."
}

if (-not (Test-ContainerRunning "s18share-redis")) {
    throw "s18share-redis is not running."
}

if ($Mode -eq "ollama") {
    if (-not (Test-ContainerRunning "s18share-ollama")) {
        throw "Mode=ollama requires s18share-ollama container."
    }

    # 1) API can reach Ollama tags endpoint.
    Invoke-InContainerPython -Container "s18share-api" -Code "import requests; r=requests.get('http://ollama:11434/api/tags', timeout=15); print('ollama_tags', r.status_code); r.raise_for_status()"

    # 2) Required models exist and support generation + embeddings paths.
    Invoke-InContainerPython -Container "s18share-api" -Code @"
import requests
base='http://ollama:11434'
tags=requests.get(base+'/api/tags', timeout=15).json()
names={m.get('name','') for m in tags.get('models',[])}
required=['gemma3:4b','nomic-embed-text']
for model in required:
    if model not in names and (model+':latest') not in names:
        raise SystemExit(f'missing_model:{model}')
g=requests.post(base+'/api/generate', json={'model':'gemma3:4b','prompt':'OK','stream':False}, timeout=60)
print('generate', g.status_code)
g.raise_for_status()
e=requests.post(base+'/api/embeddings', json={'model':'nomic-embed-text','prompt':'hello'}, timeout=30)
print('embeddings', e.status_code)
e.raise_for_status()
"@
}
else {
    # llama_cpp mode: verify endpoint health via configured base URL inside API container.
    Invoke-InContainerPython -Container "s18share-api" -Code @"
import os
import requests
base=os.environ.get('LLAMA_CPP_BASE_URL','').rstrip('/')
if not base:
    raise SystemExit('missing_env:LLAMA_CPP_BASE_URL')
r=requests.get(base+'/v1/models', timeout=20)
print('llama_cpp_models', r.status_code)
r.raise_for_status()
"@
}

# Optional cross-stack check if wise backend is up.
if (Test-ContainerRunning "wiseai-backend-1") {
    Invoke-InContainerPython -Container "wiseai-backend-1" -Code "import requests; r=requests.get('http://s18share-api:8000/health', timeout=10); print('wise_to_s18_health', r.status_code); r.raise_for_status()"
}

Write-Host "Preflight OK for mode '$Mode'."
