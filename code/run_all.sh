#!/usr/bin/env bash
# Reproduce every task end to end, in order, then build and check the submission zip.
# Usage (from the project root):  bash code/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
steps=(explore_data task1_bm25 task2_dense task3_hybrid_rerank task4_finetune task5_verification report_q1)

for step in "${steps[@]}"; do
    echo "=== ${step} ($(date +%H:%M:%S)) ==="
    "$PYTHON" "code/${step}.py"
done

rm -f 24B2232_nlp_bootcamp.zip
zip -qr 24B2232_nlp_bootcamp.zip code predictions results.json report.md -x "code/__pycache__/*"
"$PYTHON" check_format.py 24B2232_nlp_bootcamp.zip
