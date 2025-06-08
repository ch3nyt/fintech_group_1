#!/bin/bash

# Training script for VITONhd format temporal dataset
echo "=== Temporal VITONhd Training ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
DATASET_PATH="./dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_checkpoints"
EXPERIMENT_NAME="temporal_vitonhd_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Optional: Run evaluation on test set
echo "Running evaluation on test set..." | tee -a "$LOG_FILE"
python3 src/eval_temporal.py \
    --dataset_path $DATASET_PATH \
    --checkpoint_path /root/multimodal-garment-designer/temporal_vitonhd_checkpoints/temporal_vitonhd_20250531_184521/checkpoint-5000/unet.pth \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME/test_results \
    --batch_size 1 \
    --num_workers_test 1 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --no_pose True 2>&1 | tee -a "$LOG_FILE"

echo "=== All tasks completed ===" | tee -a "$LOG_FILE"