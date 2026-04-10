FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-ci.txt /app/requirements-ci.txt

RUN pip install --upgrade pip && \
    pip install uv && \
    pip install -r /app/requirements-ci.txt

COPY . /app

FROM base AS runtime

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Railway and other hosts set PORT; default 8000 for local Docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; p=os.environ.get('PORT','8000'); urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3)"

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 75"]

FROM base AS ci

# CI target: lightweight sanity check over the full source tree.
CMD ["python", "-m", "compileall", "-q", "."]
