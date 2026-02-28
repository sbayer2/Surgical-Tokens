#!/bin/bash
# Launch distributed training across multiple GPUs using accelerate.
#
# Usage:
#   bash scripts/launch_distributed.sh pretrain   # Run pretraining
#   bash scripts/launch_distributed.sh dpo        # Run DPO training
#   bash scripts/launch_distributed.sh all        # Run full pipeline
#
# Prerequisites:
#   pip install -e .
#   accelerate config  (or use the auto-detected config)

set -euo pipefail

STAGE="${1:-all}"
CONFIG="${CONFIG:-configs/default.yaml}"
OUTPUT="${OUTPUT:-./outputs}"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo "=== Surgical Tokens Distributed Training ==="
echo "Stage:    $STAGE"
echo "Config:   $CONFIG"
echo "Output:   $OUTPUT"
echo "GPUs:     $NUM_GPUS"
echo "============================================="

case "$STAGE" in
  pretrain)
    accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
      -m surgical_tokens.cli train \
      --case-log "$OUTPUT/synthetic_data/case_log.csv" \
      --tokens "$OUTPUT/codebook/tokenized_clips.pt" \
      --clip-mapping "$OUTPUT/synthetic_data/clip_mapping.json" \
      --config "$CONFIG"
    ;;

  dpo)
    accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
      -m surgical_tokens.cli train \
      --case-log "$OUTPUT/synthetic_data/case_log.csv" \
      --tokens "$OUTPUT/codebook/tokenized_clips.pt" \
      --clip-mapping "$OUTPUT/synthetic_data/clip_mapping.json" \
      --config "$CONFIG" \
      --skip-pretrain \
      --pretrained-model "$OUTPUT/models/pretrain/final"
    ;;

  all)
    echo "--- Step 1: Generate synthetic data ---"
    python -m surgical_tokens.cli generate-synthetic --output-dir "$OUTPUT/synthetic_data"

    echo "--- Step 2: Build vocabulary ---"
    python -m surgical_tokens.cli build-vocab \
      --embeddings "$OUTPUT/synthetic_data/embeddings.pt" \
      --config "$CONFIG"

    echo "--- Step 3: Train (pretrain + DPO) ---"
    accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
      -m surgical_tokens.cli train \
      --case-log "$OUTPUT/synthetic_data/case_log.csv" \
      --tokens "$OUTPUT/codebook/tokenized_clips.pt" \
      --clip-mapping "$OUTPUT/synthetic_data/clip_mapping.json" \
      --config "$CONFIG"

    echo "--- Step 4: Evaluate ---"
    python -m surgical_tokens.cli evaluate \
      --model-path "$OUTPUT/models/dpo/final" \
      --embeddings "$OUTPUT/synthetic_data/embeddings.pt" \
      --config "$CONFIG"
    ;;

  *)
    echo "Unknown stage: $STAGE"
    echo "Usage: $0 {pretrain|dpo|all}"
    exit 1
    ;;
esac

echo "=== Done ==="
