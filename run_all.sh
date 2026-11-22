#!/usr/bin/env bash
# Train all four models one after another. Safe to rerun after an interruption:
# finished models are skipped and an interrupted one resumes from checkpoints/<model>/last.pt.
set -e
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p outputs/logs
# The last two are control experiments for the Q6 analysis.
for m in temporal fixed adaptive hybrid adaptive_nomiss temporal_tod; do
    if [ -f "outputs/results/metrics_$m.json" ]; then
        echo "skip $m (done)"
        continue
    fi
    $PY train.py --model "$m" "$@" 2>&1 | tee -a "outputs/logs/train_$m.log"
done
