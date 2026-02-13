FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN python - <<'PY' && \
    pip install --upgrade pip && \
    pip install uv && \
    pip install -r /tmp/requirements.txt
import tomllib
from pathlib import Path

pyproject = Path("/app/pyproject.toml")
data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
deps = data.get("project", {}).get("dependencies", [])
Path("/tmp/requirements.txt").write_text("\n".join(deps) + "\n", encoding="utf-8")
PY

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
