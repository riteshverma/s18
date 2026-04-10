#!/bin/sh
set -e
PORT="${PORT:-8000}"
export PORT
# Visible in Railway deploy logs (unbuffered via Dockerfile PYTHONUNBUFFERED=1)
printf '%s\n' "[s18] entrypoint PORT=${PORT} pwd=$(pwd)"

exec python -m uvicorn api:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --timeout-keep-alive 75 \
  --access-log \
  --log-level info
