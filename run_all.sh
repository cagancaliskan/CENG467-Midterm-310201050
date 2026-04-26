#!/usr/bin/env bash
# =============================================================================
# run_all.sh — runs all five midterm questions sequentially
# =============================================================================

set -e   # exit on first error

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -d "ceng467_env" ]]; then
        source ceng467_env/bin/activate
    else
        echo "[ERROR] virtual environment not found. Create with:"
        echo "        python3 -m venv ceng467_env && source ceng467_env/bin/activate"
        echo "        pip install -r requirements.txt"
        exit 1
    fi
fi

START_TIME=$(date +%s)
echo "============================================================"
echo "  CENG 467 — running all five questions sequentially"
echo "  Start time: $(date)"
echo "============================================================"

run_question () {
    local dir=$1
    local script=$2
    echo ""
    echo ">>> Running $dir / $script"
    pushd "$dir" > /dev/null
    python "$script"
    popd > /dev/null
    echo "<<< Finished $dir"
}

run_question  q1_classification  q1_main.py
run_question  q2_ner             q2_main.py
run_question  q3_summarization   q3_main.py
run_question  q4_translation     q4_main.py
run_question  q5_lm              q5_main.py

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))

echo ""
echo "============================================================"
echo "  All questions completed in ${ELAPSED}s ($((ELAPSED/60)) min)"
echo "  Result files:"
echo "    q1_classification/q1_results/q1_results.json"
echo "    q2_ner/q2_results/q2_results.json"
echo "    q3_summarization/q3_results/q3_results.json"
echo "    q4_translation/q4_results/q4_results.json"
echo "    q5_lm/q5_results/q5_results.json"
echo "============================================================"