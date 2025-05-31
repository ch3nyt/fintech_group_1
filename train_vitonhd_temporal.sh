#!/bin/bash

# Training script for VITONhd format temporal dataset
echo "=== Temporal VITONhd Training ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
DATASET_PATH="/root/multimodal-garment-designer/dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_checkpoints"
EXPERIMENT_NAME="temporal_vitonhd_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Training parameters
echo "Starting training with the following configuration:" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run training with ultra-memory-efficient settings
echo "Starting training..." | tee -a "$LOG_FILE"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/train_temporal.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --num_past_weeks 4 \
    --temporal_weight_decay 0.8 \
    --temporal_loss_weight 0.3 \
    --learning_rate 1e-5 \
    --max_train_steps 10000 \
    --batch_size 1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --num_workers 1 \
    --use_wandb \
    --project_name "temporal-vitonhd" \
    --seed 42 2>&1 | tee -a "$LOG_FILE"

echo "Training completed!" | tee -a "$LOG_FILE"

# Optional: Run evaluation on test set
echo "Running evaluation on test set..." | tee -a "$LOG_FILE"
python3 src/eval_temporal.py \
    --dataset_path $DATASET_PATH \
    --checkpoint_path $OUTPUT_DIR/$EXPERIMENT_NAME/final_model/unet.pth \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME/test_results \
    --batch_size 1 \
    --num_workers_test 1 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --no_pose True 2>&1 | tee -a "$LOG_FILE"

echo "=== All tasks completed ===" | tee -a "$LOG_FILE"