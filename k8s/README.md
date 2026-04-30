# Kubernetes Deployment

This directory contains baseline manifests to run S18Share on Kubernetes.

## Prerequisites

- Built and pushed runtime image (replace in `deployment.yaml`):
  - `ghcr.io/your-org/s18share:latest`
- A Kubernetes cluster with:
  - storage class for PVCs
  - ingress controller (if using `ingress.yaml`)
- Optional for worker autoscaling:
  - [KEDA](https://keda.sh/docs/latest/deploy/) installed in the cluster

## 1) Configure secrets and image

1. Copy and edit secret template:
   - `k8s/secret.template.yaml` -> `k8s/secret.yaml`
2. Set `AZURE_OPENAI_API_KEY` (and optionally `GEMINI_API_KEY` for fallback) in `k8s/secret.yaml`.
3. Set `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`, `AZURE_OPENAI_CHAT_DEPLOYMENT`, and `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` in `k8s/configmap.yaml`.
4. Update `image:` in `k8s/deployment.yaml`.

## 2) Apply manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/pvc-memory.yaml
kubectl apply -f k8s/pvc-faiss-index.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

Enable `/runs` queue autoscaling (KEDA + Redis scaler):

```bash
kubectl apply -f k8s/keda-worker-scaledobject.yaml
```

Enable ingest queue autoscaling with a separate envelope:

```bash
kubectl apply -f k8s/keda-ingest-worker-scaledobject.yaml
```

Preset tuning values are documented in `k8s/keda-values.md`.

If Redis requires authentication:

1. Copy `k8s/keda-triggerauth.template.yaml` -> `k8s/keda-triggerauth.yaml`
2. Set `redis-password`
3. Apply it: `kubectl apply -f k8s/keda-triggerauth.yaml`
4. Uncomment `authenticationRef` in:
   - `k8s/keda-worker-scaledobject.yaml`
   - `k8s/keda-ingest-worker-scaledobject.yaml`

## 3) Verify

```bash
kubectl -n s18share get pods,svc,pvc,hpa,scaledobject
kubectl -n s18share logs deploy/s18share-api --tail=200
kubectl -n s18share describe scaledobject s18share-worker-redis
kubectl -n s18share describe scaledobject s18share-ingest-worker-redis
```

## Notes

- Start with `replicas: 1` while using file-based state (SQLite/snapshots/index files).
- Scale-out safely after migrating state to external/shared data stores.
- Set `S18_RUN_EXECUTOR=celery` for the API once Redis and the worker deployment are ready to own run execution.
- Worker split with separate queue behavior:
  - `s18share-worker` consumes queue `celery` (`/runs` tasks)
  - `s18share-ingest-worker` consumes queue `ingest` (ingest pipeline tasks)
- API autoscaling remains on `k8s/hpa.yaml` (unchanged).
- `k8s/configmap.yaml` sets strict MCP readiness defaults for `/runs`:
  - `MCP_MODE=strict`
  - `MCP_REQUIRED_SERVERS=rag,sandbox`
  - `MCP_STARTUP_TIMEOUT_SECONDS=10`
- Queue names can be adjusted via:
  - `S18_CELERY_RUNS_QUEUE`
  - `S18_CELERY_INGEST_QUEUE`
