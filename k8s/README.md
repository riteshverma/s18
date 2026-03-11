# Kubernetes Deployment

This directory contains baseline manifests to run S18Share on Kubernetes.

## Prerequisites

- Built and pushed runtime image (replace in `deployment.yaml`):
  - `ghcr.io/your-org/s18share:latest`
- A Kubernetes cluster with:
  - storage class for PVCs
  - ingress controller (if using `ingress.yaml`)

## 1) Configure secrets and image

1. Copy and edit secret template:
   - `k8s/secret.template.yaml` -> `k8s/secret.yaml`
2. Set `GEMINI_API_KEY` in `k8s/secret.yaml`.
3. Update `image:` in `k8s/deployment.yaml`.

## 2) Apply manifests

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/pvc-data.yaml
kubectl apply -f k8s/pvc-memory.yaml
kubectl apply -f k8s/pvc-faiss-index.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

## 3) Verify

```bash
kubectl -n s18share get pods,svc,pvc,hpa
kubectl -n s18share logs deploy/s18share-api --tail=200
```

## Notes

- Start with `replicas: 1` while using file-based state (SQLite/snapshots/index files).
- Scale-out safely after migrating state to external/shared data stores.
