#!/usr/bin/env python3
"""
Memory profiler for HealthyPartner stress testing (GAP-007).

Polls /debug/stats every SAMPLE_INTERVAL seconds while the server is under load
and writes a CSV + a human-readable pass/fail report.

Usage:
    python tests/load/profile_memory.py [--duration 300] [--interval 30] \
                                         [--host http://localhost:8000] \
                                         [--out tests/load/results]

Pass criteria (GAP-007):
  - Peak RSS < 2 GB
  - RSS growth < 200 MB over the run (no sustained leak)
  - response_cache size never exceeds 500 (its configured max)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import urllib.request as _req
    import urllib.error as _err
except ImportError:
    raise SystemExit("Python 3 stdlib required")

# ── Pass / fail thresholds ─────────────────────────────────────────────────────

PEAK_RSS_LIMIT_MB = 2048          # hard OOM risk threshold
RSS_GROWTH_LIMIT_MB = 200         # max acceptable growth over the full run
CACHE_MAX_SIZE = 500              # _ResponseCache max_size from orchestrator.py


# ── Sampler ────────────────────────────────────────────────────────────────────


def _fetch_stats(host: str) -> Optional[dict]:
    try:
        with _req.urlopen(f"{host}/debug/stats", timeout=5) as r:
            return json.loads(r.read())
    except (_err.URLError, OSError, json.JSONDecodeError):
        return None


def run(
    host: str,
    duration_s: int,
    interval_s: int,
    out_dir: str,
) -> bool:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(out_dir) / f"memory_{ts}.csv"
    report_path = Path(out_dir) / f"memory_report_{ts}.txt"

    samples: list[dict] = []
    deadline = time.time() + duration_s

    print(f"Memory profiler started — {duration_s}s run, sampling every {interval_s}s")
    print(f"  Host:   {host}")
    print(f"  Output: {csv_path}")
    print()

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["elapsed_s", "rss_mb", "vms_mb", "session_count", "cache_default"])

        while time.time() < deadline:
            t0 = time.time()
            stats = _fetch_stats(host)
            elapsed = round(t0 - (deadline - duration_s), 1)

            if stats is None:
                print(f"  [{elapsed:>6}s] server unreachable — skipping sample")
                time.sleep(interval_s)
                continue

            rss = stats.get("rss_mb", 0)
            vms = stats.get("vms_mb", 0)
            sessions = stats.get("session_count", 0)
            caches = stats.get("tenant_caches", {})
            cache_default = caches.get("default", {}).get("response_cache", -1)

            row = [elapsed, rss, vms, sessions, cache_default]
            writer.writerow(row)
            fh.flush()
            samples.append({
                "elapsed_s": elapsed,
                "rss_mb": rss,
                "vms_mb": vms,
                "session_count": sessions,
                "cache_default": cache_default,
            })
            print(
                f"  [{elapsed:>6}s] RSS={rss:.1f} MB  "
                f"sessions={sessions}  cache={cache_default}"
            )
            time.sleep(max(0, interval_s - (time.time() - t0)))

    # ── Generate report ────────────────────────────────────────────────────────

    if not samples:
        print("\nNo samples collected — server may not have been running.")
        return False

    rss_values = [s["rss_mb"] for s in samples]
    baseline_rss = rss_values[0]
    peak_rss = max(rss_values)
    final_rss = rss_values[-1]
    rss_growth = final_rss - baseline_rss
    cache_values = [s["cache_default"] for s in samples if s["cache_default"] >= 0]
    peak_cache = max(cache_values) if cache_values else -1

    checks = [
        ("Peak RSS < 2 GB",
         peak_rss < PEAK_RSS_LIMIT_MB,
         f"{peak_rss:.1f} MB (limit {PEAK_RSS_LIMIT_MB} MB)"),
        ("RSS growth < 200 MB (no leak)",
         rss_growth < RSS_GROWTH_LIMIT_MB,
         f"+{rss_growth:.1f} MB over run (limit {RSS_GROWTH_LIMIT_MB} MB)"),
        ("Response cache within capacity",
         peak_cache <= CACHE_MAX_SIZE,
         f"peak size {peak_cache} (max {CACHE_MAX_SIZE})"),
    ]

    passed = all(c[1] for c in checks)

    lines = [
        "=" * 60,
        "HealthyPartner Memory Profile Report",
        f"Run:       {ts}",
        f"Duration:  {duration_s}s  |  Samples: {len(samples)}",
        f"Baseline RSS: {baseline_rss:.1f} MB",
        f"Peak RSS:     {peak_rss:.1f} MB",
        f"Final RSS:    {final_rss:.1f} MB",
        f"RSS growth:   {rss_growth:+.1f} MB",
        f"Peak cache:   {peak_cache}",
        "",
        "Pass/Fail",
        "-" * 40,
    ]
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"  [{status}] {label}")
        lines.append(f"         {detail}")
    lines += ["", "=" * 60, f"Overall: {'PASS' if passed else 'FAIL'}"]

    report = "\n".join(lines)
    print("\n" + report)
    report_path.write_text(report + "\n")
    print(f"\nReport saved: {report_path}")

    return passed


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="HealthyPartner memory profiler")
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--duration", type=int, default=300,
                        help="Profiling duration in seconds (default: 300)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Sample interval in seconds (default: 30)")
    parser.add_argument("--out", default="tests/load/results",
                        help="Output directory for CSV + report")
    args = parser.parse_args()

    passed = run(
        host=args.host,
        duration_s=args.duration,
        interval_s=args.interval,
        out_dir=args.out,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
