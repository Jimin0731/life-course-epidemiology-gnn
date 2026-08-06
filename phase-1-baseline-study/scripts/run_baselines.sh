#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PHASE_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PHASE_DIR/scripts/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PHASE_DIR/results/model-results}"

N_SIZES=(500 1000 2000 5000 10000 20000)
DATA_NAMES=(Repeats1 Repeats2 Repeats3 Order1 Order2 Order3 Timing1 Timing2 Timing3 Timing4 Period1 Period2 Period3 Period4)
MODELS=(LR XGBoost ResNet)

for N_VALUE in "${N_SIZES[@]}"; do
  DATA_DIR="$DATA_ROOT/data_$N_VALUE"
  if [[ ! -d "$DATA_DIR" ]]; then
    echo "Skipping N=$N_VALUE; data directory not found: $DATA_DIR"
    continue
  fi
  for DATA_NAME in "${DATA_NAMES[@]}"; do
    for MODEL_NAME in "${MODELS[@]}"; do
      echo "Running N=$N_VALUE, rule=$DATA_NAME, model=$MODEL_NAME"
      PYTHONPATH="$PHASE_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m lce \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --data-name "$DATA_NAME" \
        --model-name "$MODEL_NAME" \
        --data-split-seed 43 || true
    done
  done
done

echo "Baseline grid finished."
