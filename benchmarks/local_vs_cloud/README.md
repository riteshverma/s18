# Local-First Benchmark Harness

Use this harness to compare S18 latency across deployment profiles
(for example local Ollama vs cloud-hosted model APIs).

## What it measures

- request success rate
- end-to-end latency per run
- p50 / p95 latency summary

## Run

```bash
python benchmarks/local_vs_cloud/benchmark_runs.py \
  --base-url http://localhost:8000 \
  --profile local-laptop-gemma \
  --scenario-file benchmarks/local_vs_cloud/scenarios.json \
  --iterations 2
```

Then run the same command against your cloud endpoint (change `--profile` label).

## Output

Results are written to:
- `benchmarks/local_vs_cloud/results/<profile>_<timestamp>.json`

Use these files to publish privacy-first and cost-aware benchmark reports.
