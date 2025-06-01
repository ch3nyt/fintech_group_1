#!/bin/bash

# DPO Inference script for VITONhd format temporal dataset
echo "=== DPO-Enhanced Temporal VITONhd Inference ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
DATASET_PATH="/root/multimodal-garment-designer/dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_dpo_inference"
EXPERIMENT_NAME="temporal_vitonhd_dpo_inference_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/inference.log"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Inference parameters
echo "Starting DPO inference with the following configuration:" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Display DPO-specific parameters
echo "" | tee -a "$LOG_FILE"
echo "DPO Inference Parameters:" | tee -a "$LOG_FILE"
echo "- Number of inference steps: 50" | tee -a "$LOG_FILE"
echo "- Guidance scale: 7.5" | tee -a "$LOG_FILE"
echo "- Mixed precision: fp16" | tee -a "$LOG_FILE"
echo "- Batch size: 1" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find ./temporal_vitonhd_dpo_checkpoints -name "checkpoint-*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found, using final model..." | tee -a "$LOG_FILE"
    CHECKPOINT_PATH="./temporal_vitonhd_dpo_checkpoints/final_model/unet.pth"
else
    echo "Using checkpoint: $LATEST_CHECKPOINT" | tee -a "$LOG_FILE"
    CHECKPOINT_PATH="$LATEST_CHECKPOINT/unet.pth"
fi

# Run DPO inference
echo "Starting DPO inference..." | tee -a "$LOG_FILE"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/eval_temporal.py \
    --dataset_path $DATASET_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --num_past_weeks 8 \
    --temporal_weight_decay 0.8 \
    --batch_size 1 \
    --num_workers_test 1 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --mixed_precision fp16 \
    --no_pose True 2>&1 | tee -a "$LOG_FILE"

echo "DPO Inference completed!" | tee -a "$LOG_FILE"

# Print summary statistics
echo "" | tee -a "$LOG_FILE"
echo "Inference Summary:" | tee -a "$LOG_FILE"
echo "- Inference logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "- Generated images: $OUTPUT_DIR/$EXPERIMENT_NAME/predictions" | tee -a "$LOG_FILE"
echo "- Caption log: $OUTPUT_DIR/$EXPERIMENT_NAME/captions_used.txt" | tee -a "$LOG_FILE"
echo "- Metadata: $OUTPUT_DIR/$EXPERIMENT_NAME/predictions_metadata.json" | tee -a "$LOG_FILE"
echo "- Category stats: $OUTPUT_DIR/$EXPERIMENT_NAME/category_statistics.json" | tee -a "$LOG_FILE"

# Optional: Generate a quick visualization of results
echo "" | tee -a "$LOG_FILE"
echo "Generating visualization of results..." | tee -a "$LOG_FILE"
python3 -c "
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load category statistics
stats_path = '$OUTPUT_DIR/$EXPERIMENT_NAME/category_statistics.json'
if os.path.exists(stats_path):
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # Create bar plot
    categories = list(stats.keys())
    counts = list(stats.values())

    plt.figure(figsize=(12, 6))
    plt.bar(categories, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('DPO Inference Results by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of Predictions')
    plt.tight_layout()

    # Save plot
    plt.savefig('$OUTPUT_DIR/$EXPERIMENT_NAME/category_distribution.png')
    print('✅ Generated category distribution plot')
else:
    print('❌ Category statistics file not found')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== DPO Inference Completed ===" | tee -a "$LOG_FILE"
echo "Results saved in: $OUTPUT_DIR/$EXPERIMENT_NAME" | tee -a "$LOG_FILE"