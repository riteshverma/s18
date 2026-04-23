"""Simple benchmark harness for comparing local vs cloud S18 runtimes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    rank = int(round((percentile / 100) * (len(sorted_values) - 1)))
    rank = max(0, min(rank, len(sorted_values) - 1))
    return sorted_values[rank]


def _run_once(base_url: str, payload: dict, timeout_seconds: int) -> tuple[bool, float, int]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url.rstrip('/')}/runs",
        json=payload,
        timeout=timeout_seconds,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return response.ok, elapsed_ms, response.status_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /runs latency across profiles.")
    parser.add_argument("--base-url", required=True, help="S18 base URL, e.g. http://localhost:8000")
    parser.add_argument("--profile", required=True, help="Label for benchmark run (local/cloud profile name)")
    parser.add_argument("--scenario-file", required=True, help="Path to scenario JSON array")
    parser.add_argument("--iterations", type=int, default=1, help="Iterations per scenario")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="HTTP timeout for each run")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenario_file).read_text(encoding="utf-8"))
    records: list[dict] = []
    latencies: list[float] = []

    for scenario in scenarios:
        scenario_name = scenario["name"]
        payload = scenario["payload"]
        for iteration in range(1, args.iterations + 1):
            ok, elapsed_ms, status_code = _run_once(args.base_url, payload, args.timeout_seconds)
            records.append(
                {
                    "scenario": scenario_name,
                    "iteration": iteration,
                    "ok": ok,
                    "status_code": status_code,
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            if ok:
                latencies.append(elapsed_ms)
            print(f"{scenario_name} [{iteration}/{args.iterations}] -> {status_code} in {elapsed_ms:.2f} ms")

    sorted_latencies = sorted(latencies)
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "base_url": args.base_url,
        "iterations": args.iterations,
        "scenario_count": len(scenarios),
        "total_runs": len(records),
        "success_runs": len(latencies),
        "success_rate": round((len(latencies) / len(records)) * 100, 2) if records else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(sorted_latencies), 2) if sorted_latencies else 0.0,
            "p50": round(_percentile(sorted_latencies, 50), 2),
            "p95": round(_percentile(sorted_latencies, 95), 2),
            "min": round(min(sorted_latencies), 2) if sorted_latencies else 0.0,
            "max": round(max(sorted_latencies), 2) if sorted_latencies else 0.0,
        },
    }
    output = {"summary": summary, "records": records}

    results_dir = Path("benchmarks/local_vs_cloud/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"{args.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Benchmark results saved: {output_file}")


if __name__ == "__main__":
    main()
