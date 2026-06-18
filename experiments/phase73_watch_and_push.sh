#!/usr/bin/env bash
# Detached watcher: waits for the Phase 73 run to finish, then auto-generates a
# phone-readable results doc and pushes it to GitHub. Survives this Claude session
# closing (launch with: nohup bash experiments/phase73_watch_and_push.sh >phase73_watch.log 2>&1 &).
set -u
cd "$(dirname "$0")/.."   # repo root
LOG=phase73_run.log
echo "[watch] started $(date)"

STATUS=timeout
for i in $(seq 1 900); do            # up to ~30h (900 * 120s) — ETA grew to ~16h, headroom for safety
  if grep -q "Results saved to phase73_output.json" "$LOG" 2>/dev/null; then
    STATUS=complete; break
  fi
  # If no python is running the marker never appeared -> run ended/crashed.
  if ! tasklist 2>/dev/null | grep -qi 'python.exe'; then
    # give the final incremental write a moment, then finalize on whatever we have
    sleep 5
    if grep -q "Results saved to phase73_output.json" "$LOG" 2>/dev/null; then STATUS=complete; else STATUS=ended_no_marker; fi
    break
  fi
  sleep 120
done
echo "[watch] phase73 status=$STATUS $(date)"

python experiments/phase73_autofinalize.py || { echo "[watch] autofinalize failed"; exit 1; }

git add docs/phase_73_results.md
if git diff --cached --quiet; then
  echo "[watch] nothing to commit"; exit 0
fi
git commit -m "Phase 73 (auto): high-power edge-sampled hop ablation results [$STATUS]

Auto-finalized snapshot generated from phase73_output.json on run completion, so
results are viewable remotely (phone) without an interactive session. A polished
docs/phase_73.md + research_state.json update follow in an interactive session.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin main && echo "[watch] pushed $(date)" || echo "[watch] push FAILED $(date)"
