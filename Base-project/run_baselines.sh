#!/bin/bash

# DATA_NAMES=("Repeats1" "Order1" "Timing1" "Period1") # 우선 4개만 테스트
DATA_NAMES=("Repeats1" "Repeats2" "Repeats3" "Order1" "Order2" "Order3" "Timing1" "Timing2" "Timing3" "Timing4" "Period1" "Period2" "Period3" "Period4")

MODELS=("LR" "XGBoost" "ResNet") 
# MODELS=("LR" "XGBoost" "ResNet" "LSTMAttention" "MLSTMFCN" "InceptionTime")

for DATA in "${DATA_NAMES[@]}"
do
  for MODEL in "${MODELS[@]}"
  do
    echo "========================================================"
    echo "Running: $MODEL on $DATA"
    echo "========================================================"
    
    lce --data-name "$DATA" --model-name "$MODEL" --data-split-seed 43 || true
    
    echo "Finished: $MODEL on $DATA"
  done
done

echo "All baseline runs complete."
