# KEDA Preset Values (`/runs` vs ingest)

This file gives quick starting presets for Redis-list KEDA scaling in S18Share:

- `/runs` worker: `k8s/keda-worker-scaledobject.yaml` (`listName: celery`)
- ingest worker: `k8s/keda-ingest-worker-scaledobject.yaml` (`listName: ingest`)

Tune by editing `listLength`, `maxReplicaCount`, and `cooldownPeriod`.

## Presets

| Traffic profile | `/runs` `listLength` | `/runs` `maxReplicaCount` | `/runs` `cooldownPeriod` | ingest `listLength` | ingest `maxReplicaCount` | ingest `cooldownPeriod` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Low | 3 | 4 | 60 | 1 | 2 | 120 |
| Medium | 5 | 8 | 60 | 2 | 4 | 120 |
| High | 8 | 16 | 45 | 4 | 8 | 90 |

## How to pick a profile

- **Low**: early pilots, small tenant count, mostly daytime usage.
- **Medium**: default recommendation for multi-tenant staging/prod.
- **High**: bursty workloads, heavy document ingest windows, strict queue-latency SLOs.

## Quick tuning rules

- If queue latency is too high, decrease `listLength` or increase `maxReplicaCount`.
- If pods flap too often, increase `cooldownPeriod`.
- Keep ingest scaling more conservative than `/runs` unless your workload is ingest-heavy.
- Keep API scaling on `k8s/hpa.yaml` (no KEDA change required for API in current setup).

## Copy-paste YAML edits

Apply one preset by updating these fields in both ScaledObjects.

### Low

```yaml
# k8s/keda-worker-scaledobject.yaml
cooldownPeriod: 60
maxReplicaCount: 4
triggers:
  - metadata:
      listLength: "3"

# k8s/keda-ingest-worker-scaledobject.yaml
cooldownPeriod: 120
maxReplicaCount: 2
triggers:
  - metadata:
      listLength: "1"
```

### Medium

```yaml
# k8s/keda-worker-scaledobject.yaml
cooldownPeriod: 60
maxReplicaCount: 8
triggers:
  - metadata:
      listLength: "5"

# k8s/keda-ingest-worker-scaledobject.yaml
cooldownPeriod: 120
maxReplicaCount: 4
triggers:
  - metadata:
      listLength: "2"
```

### High

```yaml
# k8s/keda-worker-scaledobject.yaml
cooldownPeriod: 45
maxReplicaCount: 16
triggers:
  - metadata:
      listLength: "8"

# k8s/keda-ingest-worker-scaledobject.yaml
cooldownPeriod: 90
maxReplicaCount: 8
triggers:
  - metadata:
      listLength: "4"
```
