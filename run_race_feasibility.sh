#!/usr/bin/env bash

# ============================================================
# RACE Minimal Reproduction Feasibility Test
# ============================================================
# Purpose:
#   1. Verify the Python environment
#   2. Verify key RACE dependencies
#   3. Run a 10,000-step RACE/SMAC training test
#   4. Capture stdout/stderr
#   5. Record PASS/FAIL evidence
#
# Expected working directory:
#   RACE repository root
# ============================================================

set -o pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

set -a
source .env
set +a

# Add SMAC to Python's import path
export PYTHONPATH="$PYTHONPATH"

# Check the environment
python3 -c "import smac; print('SMAC:', smac.__file__)"
echo "SC2PATH=$SC2PATH"


TEST_NAME="RACE_10K_FEASIBILITY"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

RESULT_DIR="feasibility_results/${TIMESTAMP}"
LOG_FILE="${RESULT_DIR}/training.log"
SUMMARY_FILE="${RESULT_DIR}/summary.txt"
ENV_FILE="${RESULT_DIR}/environment.txt"

mkdir -p "$RESULT_DIR"

PASS_COUNT=0
FAIL_COUNT=0

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

pass() {
    echo "[PASS] $1" | tee -a "$SUMMARY_FILE"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo "[FAIL] $1" | tee -a "$SUMMARY_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

cat > "$SUMMARY_FILE" <<EOF
RACE Minimal Reproduction Feasibility Test
==========================================

Test:       $TEST_NAME
Timestamp:  $TIMESTAMP
Repository: $(pwd)

EOF

section "C0 - Repository Check"

if [[ -f "src/main.py" ]]; then
    pass "RACE src/main.py found"
else
    fail "RACE src/main.py not found"
    echo "Run this script from the RACE repository root."
    exit 1
fi

# ------------------------------------------------------------
# C1 - Environment verification
# ------------------------------------------------------------

section "C1 - Python Environment"

{
    echo "===== SYSTEM ====="
    date
    uname -a

    echo
    echo "===== PYTHON ====="
    python3 --version
    which python3

    echo
    echo "===== PIP ====="
    python3 -m pip --version

    echo
    echo "===== PYTORCH ====="
    python3 -c "import torch; print('PyTorch:', torch.__version__)"

    echo
    echo "===== NUMPY ====="
    python3 -c "import numpy; print('NumPy:', numpy.__version__)"

    echo
    echo "===== PYSC2 ====="
    python3 -c "import pysc2; print('PySC2:', pysc2.__file__)"

} > "$ENV_FILE" 2>&1

if python3 -c "import torch" >/dev/null 2>&1; then
    pass "PyTorch import successful"
else
    fail "PyTorch import failed"
fi

if python3 -c "import numpy" >/dev/null 2>&1; then
    pass "NumPy import successful"
else
    fail "NumPy import failed"
fi

if python3 -c "import pysc2" >/dev/null 2>&1; then
    pass "PySC2 import successful"
else
    fail "PySC2 import failed"
fi

# ------------------------------------------------------------
# C2 - RACE configuration check
# ------------------------------------------------------------

section "C2 - RACE Configuration"

if [[ -f "src/main.py" ]]; then
    pass "RACE main entry point exists"
else
    fail "RACE main entry point missing"
fi

if grep -R "frac" src/ >/dev/null 2>&1; then
    pass "Mutation parameter 'frac' found in RACE source"
else
    fail "Mutation parameter 'frac' not found"
fi

if grep -R "EA_freq" src/ >/dev/null 2>&1; then
    pass "Evolution frequency parameter 'EA_freq' found"
else
    fail "Evolution frequency parameter 'EA_freq' not found"
fi

# ------------------------------------------------------------
# C3 - 10,000-step training test
# ------------------------------------------------------------

section "C3 - 10,000-Step Training Test"

echo "Training command:" | tee -a "$SUMMARY_FILE"

cat <<'EOF' | tee -a "$SUMMARY_FILE"
python3 src/main.py \
  --config=facmac_smac \
  --env-config=sc2 \
  with \
  env_args.map_name=3s_vs_3z \
  batch_size_run=1 \
  state_alpha=0.001 \
  frac=0.005 \
  EA_alpha=1.0 \
  Org_alpha=1.0 \
  EA=1 \
  EA_freq=1 \
  SAME=0 \
  use_cuda=False \
  t_max=10000
EOF

echo
echo "Starting training..."
echo "Full training output will be saved to:"
echo "$LOG_FILE"
echo

START_TIME=$(date +%s)

python3 src/main.py \
  --config=facmac_smac \
  --env-config=sc2 \
  with \
  env_args.map_name=3s_vs_3z \
  batch_size_run=1 \
  state_alpha=0.001 \
  frac=0.005 \
  EA_alpha=1.0 \
  Org_alpha=1.0 \
  EA=1 \
  EA_freq=1 \
  SAME=0 \
  use_cuda=False \
  t_max=10000 \
  2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo
echo "Training exit code: $TRAIN_EXIT_CODE" | tee -a "$SUMMARY_FILE"
echo "Training duration: ${ELAPSED} seconds" | tee -a "$SUMMARY_FILE"

if [[ "$TRAIN_EXIT_CODE" -eq 0 ]]; then
    pass "10,000-step RACE training process exited successfully"
else
    fail "10,000-step RACE training process exited with code $TRAIN_EXIT_CODE"
fi

# ------------------------------------------------------------
# C4 - Evidence that training actually progressed
# ------------------------------------------------------------

section "C4 - Training Progress Evidence"

# Search the log for common indicators of actual training.
PROGRESS_FOUND=0

if grep -Eiq "episode|return|reward|t_env|train" "$LOG_FILE"; then
    PROGRESS_FOUND=1
fi

if [[ "$PROGRESS_FOUND" -eq 1 ]]; then
    pass "Training log contains evidence of training/episode progression"
else
    fail "Could not find recognizable training progression in log"
fi

# Look for a final/terminal training indicator.
if grep -Eiq "Finished Training|finished|complete|return|episode" "$LOG_FILE"; then
    pass "Training log contains terminal/progress indicators"
else
    fail "No recognizable terminal/progress indicator found"
fi

# ------------------------------------------------------------
# C5 - Output/result evidence
# ------------------------------------------------------------

section "C5 - Output Evidence"

# Search for recently created files/directories.
OUTPUT_FILES=$(find . -type f -newermt "@$START_TIME" 2>/dev/null | head -50)

if [[ -n "$OUTPUT_FILES" ]]; then
    pass "Files were created/modified during the training run"

    echo
    echo "Recently created/modified files:"
    echo "$OUTPUT_FILES" | tee -a "$SUMMARY_FILE"
else
    fail "No files appear to have been created/modified during training"
fi

# ------------------------------------------------------------
# Save git state
# ------------------------------------------------------------

section "Repository State"

{
    echo
    echo "===== GIT COMMIT ====="
    git rev-parse HEAD 2>/dev/null || true

    echo
    echo "===== GIT STATUS ====="
    git status --short 2>/dev/null || true
} >> "$ENV_FILE"

# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

section "FINAL RESULT"

TOTAL=$((PASS_COUNT + FAIL_COUNT))

echo
echo "PASS: $PASS_COUNT"
echo "FAIL: $FAIL_COUNT"
echo "TOTAL CHECKS: $TOTAL"
echo

if [[ "$FAIL_COUNT" -eq 0 ]]; then
    echo "OVERALL RESULT: PASS" | tee -a "$SUMMARY_FILE"
    echo
    echo "RACE passed the minimal reproduction feasibility test." \
        | tee -a "$SUMMARY_FILE"
    EXIT_STATUS=0
else
    echo "OVERALL RESULT: FAIL" | tee -a "$SUMMARY_FILE"
    echo
    echo "RACE did not pass the minimal reproduction feasibility test." \
        | tee -a "$SUMMARY_FILE"
    EXIT_STATUS=1
fi

echo
echo "Evidence directory:"
echo "  $RESULT_DIR"

echo
echo "Files:"
echo "  Training log : $LOG_FILE"
echo "  Summary      : $SUMMARY_FILE"
echo "  Environment  : $ENV_FILE"

exit "$EXIT_STATUS"
