#!/usr/bin/env bash
# Detached watcher: push Phase 76 DIAGNOSE_5p results to GitHub on completion.
set -u; cd "$(dirname "$0")/.."
for i in $(seq 1 600); do
  grep -q "DONE. saved" phase76_run.log 2>/dev/null && break
  tasklist 2>/dev/null | grep -qi 'python.exe' || { sleep 5; break; }
  sleep 60
done
git add -f phase76_output.json 2>/dev/null
git diff --cached --quiet && { echo "[watch] nothing to commit"; exit 0; }
git commit -q -m "Phase 76: DIAGNOSE_5p results (auto-pushed on completion)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -q origin main && echo "[watch] pushed $(date)"
