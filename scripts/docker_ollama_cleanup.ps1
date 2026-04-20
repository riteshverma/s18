# Free disk: Docker build cache + unused images + Ollama unused layers (safe for named volumes).
# Run in PowerShell:  .\scripts\docker_ollama_cleanup.ps1

$ErrorActionPreference = "Continue"
Write-Host "=== Docker build cache ===" -ForegroundColor Cyan
docker builder prune -af

Write-Host "`n=== Docker unused images (not used by any container) ===" -ForegroundColor Cyan
docker image prune -a -f

Write-Host "`n=== Docker dangling resources ===" -ForegroundColor Cyan
docker system prune -f

Write-Host "`n=== Ollama (Docker): prune unused blobs if supported ===" -ForegroundColor Cyan
$cid = docker ps -q --filter "name=s18share-ollama"
if (-not $cid) { $cid = docker ps -q --filter "ancestor=ollama/ollama" }
if ($cid) {
    docker exec $cid ollama prune 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "If 'unknown command', your Ollama image is older; remove models with: docker exec -it <container> ollama rm <model>"
    }
} else {
    Write-Host "No running Ollama container. For native Windows Ollama:  ollama list  then  ollama rm <unused-model>"
}

Write-Host "`n=== Disk usage after (docker system df) ===" -ForegroundColor Cyan
docker system df

Write-Host "`nDone." -ForegroundColor Green
