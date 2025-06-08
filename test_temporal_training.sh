#!/bin/bash

# Quick test script for temporal training to verify everything works
echo "=== Testing Temporal VITONhd Training ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
DATASET_PATH="./dataset_vitonhd_format"
OUTPUT_DIR="./test_temporal_checkpoints"
EXPERIMENT_NAME="test_temporal_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

echo "Starting test training with the following configuration:" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Run a quick test training (only 10 steps)
echo "Starting test training..." | tee -a "$LOG_FILE"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/train_temporal.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --num_past_weeks 2 \
    --temporal_weight_decay 0.8 \
    --temporal_loss_weight 0.3 \
    --learning_rate 1e-5 \
    --max_train_steps 10 \
    --batch_size 1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 2 \
    --save_steps 5 \
    --num_workers 1 \
    --project_name "test-temporal-vitonhd" \
    --seed 42 2>&1 | tee -a "$LOG_FILE"

echo "Test training completed!" | tee -a "$LOG_FILE"

# Check if model was saved
if [ -f "$OUTPUT_DIR/$EXPERIMENT_NAME/final_model/unet.pth" ]; then
    echo "✅ Model saved successfully" | tee -a "$LOG_FILE"
else
    echo "❌ Model not found" | tee -a "$LOG_FILE"
fi

echo "=== Test completed ===" | tee -a "$LOG_FILE"