#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# GAP-007 Stress test runner for HealthyPartner v2
#
# Runs locust + memory profiler in parallel against a live server.
# Server must already be running on port 8000.
#
# Usage:
#   ./tests/load/run_stress_test.sh [--users N] [--duration Xs] [--host URL]
#
# Defaults:
#   users    = 50
#   duration = 5m   (use 30m for the full GAP-007 endurance test)
#   host     = http://localhost:8000
#
# Pass criteria (GAP-007):
#   - Locust error rate < 1%
#   - P95 < 3s  for /health, /admin/kb/stats
#   - P95 < 8s  for /chat
#   - No OOM (peak RSS < 2 GB, growth < 200 MB) — from memory profiler
# ────────────────────────────────────────────────────────────────

set -euo pipefail

USERS=50
DURATION="5m"
HOST="http://localhost:8000"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --users)    USERS="$2";    shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --host)     HOST="$2";     shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# ── Pre-flight check ──────────────────────────────────────────────────────────
echo "── Pre-flight check ──────────────────────────────────────────"
if ! curl -sf "$HOST/health" > /dev/null 2>&1; then
  echo "ERROR: Server not reachable at $HOST"
  echo "Start it with:  uvicorn app.main:app --port 8000"
  exit 1
fi
echo "Server OK: $HOST"
echo ""

# ── Convert duration string to seconds for profile_memory.py ─────────────────
duration_s() {
  local d="$1"
  if [[ "$d" =~ ^([0-9]+)m$ ]]; then echo $(( ${BASH_REMATCH[1]} * 60 ))
  elif [[ "$d" =~ ^([0-9]+)s$ ]]; then echo "${BASH_REMATCH[1]}"
  elif [[ "$d" =~ ^([0-9]+)h$ ]]; then echo $(( ${BASH_REMATCH[1]} * 3600 ))
  else echo 300  # fallback
  fi
}
DURATION_S=$(duration_s "$DURATION")

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi

# ── Run locust headless (background) ─────────────────────────────────────────
echo "── Locust ($USERS users, $DURATION) ──────────────────────────"
LOCUST_CSV="$RESULTS_DIR/locust_${TS}"
locust \
  -f "$SCRIPT_DIR/locustfile.py" \
  --host "$HOST" \
  --users "$USERS" \
  --spawn-rate 5 \
  --run-time "$DURATION" \
  --headless \
  --csv "$LOCUST_CSV" \
  --csv-full-history \
  --only-summary \
  > "$RESULTS_DIR/locust_stdout_${TS}.txt" 2>&1 &
LOCUST_PID=$!
echo "  PID: $LOCUST_PID  |  CSV: ${LOCUST_CSV}_stats.csv"

# ── Run memory profiler in parallel ──────────────────────────────────────────
echo ""
echo "── Memory profiler ($DURATION_S s) ───────────────────────────"
python "$SCRIPT_DIR/profile_memory.py" \
  --host "$HOST" \
  --duration "$DURATION_S" \
  --interval 30 \
  --out "$RESULTS_DIR" &
MEM_PID=$!

# ── Wait for both to finish ───────────────────────────────────────────────────
echo ""
echo "Both tools running — waiting for completion..."
LOCUST_EXIT=0; MEM_EXIT=0
wait "$LOCUST_PID" || LOCUST_EXIT=$?
wait "$MEM_PID"    || MEM_EXIT=$?

# ── Parse locust results ──────────────────────────────────────────────────────
echo ""
echo "── Locust summary ────────────────────────────────────────────"
STATS_FILE="${LOCUST_CSV}_stats.csv"
if [[ -f "$STATS_FILE" ]]; then
  # Print the aggregated row from the CSV
  python3 - "$STATS_FILE" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path) as f:
    rows = list(csv.DictReader(f))
# Find the Aggregated row
agg = next((r for r in rows if r.get("Name") == "Aggregated"), rows[-1] if rows else None)
if agg:
    p95 = float(agg.get("95%", 0))
    p99 = float(agg.get("99%", 0))
    failures = int(agg.get("Failure Count", 0))
    requests = int(agg.get("Request Count", 0))
    err_rate = failures / requests * 100 if requests else 0
    print(f"  Requests: {requests}  Failures: {failures}  Error rate: {err_rate:.2f}%")
    print(f"  P95: {p95}ms  P99: {p99}ms")
    locust_pass = err_rate < 1.0
    print(f"  Locust: {'PASS' if locust_pass else 'FAIL'}")
else:
    print("  (no aggregated row found)")
PYEOF
else
  echo "  (no stats CSV found — locust may have failed)"
fi

cat "$RESULTS_DIR/locust_stdout_${TS}.txt" | grep -E "(Requests|Failures|P95|P99|Aggregated)" || true

# ── Final verdict ─────────────────────────────────────────────────────────────
echo ""
echo "── Final verdict ─────────────────────────────────────────────"
OVERALL_PASS=true
[[ "$LOCUST_EXIT" -ne 0 ]] && { echo "  [FAIL] Locust exited with code $LOCUST_EXIT"; OVERALL_PASS=false; }
[[ "$MEM_EXIT"    -ne 0 ]] && { echo "  [FAIL] Memory profiler failed"; OVERALL_PASS=false; }
$OVERALL_PASS && echo "  [PASS] All checks passed" || echo "  [FAIL] One or more checks failed"

echo ""
echo "Results written to: $RESULTS_DIR"
$OVERALL_PASS && exit 0 || exit 1
