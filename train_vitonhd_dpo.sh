#!/bin/bash

# DPO Training script for VITONhd format temporal dataset
echo "=== DPO-Enhanced Temporal VITONhd Training ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
DATASET_PATH="/root/multimodal-garment-designer/dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_dpo_checkpoints"
EXPERIMENT_NAME="temporal_vitonhd_dpo_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"
mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"

# Training parameters
echo "Starting DPO training with the following configuration:" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Display DPO-specific parameters
echo "" | tee -a "$LOG_FILE"
echo "DPO Parameters:" | tee -a "$LOG_FILE"
echo "- Number of candidates: 20" | tee -a "$LOG_FILE"
echo "- DPO beta: 0.1" | tee -a "$LOG_FILE"
echo "- DPO weight: 0.5" | tee -a "$LOG_FILE"
echo "- CLIP-I weight: 0.6" | tee -a "$LOG_FILE"
echo "- CLIP-T weight: 0.4" | tee -a "$LOG_FILE"
echo "- DPO frequency: 1%" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run DPO training with memory-efficient settings
echo "Starting DPO training..." | tee -a "$LOG_FILE"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/train_vitonhd_dpo.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --num_past_weeks 8 \
    --temporal_weight_decay 0.8 \
    --temporal_loss_weight 0.3 \
    --num_candidates 20 \
    --dpo_beta 0.1 \
    --dpo_weight 0.5 \
    --clip_i_weight 0.6 \
    --clip_t_weight 0.4 \
    --dpo_frequency 0.01 \
    --num_inference_steps 20 \
    --learning_rate 1e-5 \
    --max_train_steps 1000 \
    --batch_size 1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 16 \
    --save_steps 50 \
    --num_workers 1 \
    --project_name "temporal-vitonhd-dpo" \
    --seed 42 2>&1 | tee -a "$LOG_FILE"

echo "DPO Training completed!" | tee -a "$LOG_FILE"

# Optional: Run evaluation on test set with the best checkpoint
echo "" | tee -a "$LOG_FILE"
echo "Running evaluation on test set with DPO-trained model..." | tee -a "$LOG_FILE"

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find $OUTPUT_DIR/$EXPERIMENT_NAME -name "checkpoint-*" -type d | sort -V | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found, using final model..." | tee -a "$LOG_FILE"
    CHECKPOINT_PATH="$OUTPUT_DIR/$EXPERIMENT_NAME/final_model/unet.pth"
else
    echo "Using checkpoint: $LATEST_CHECKPOINT" | tee -a "$LOG_FILE"
    CHECKPOINT_PATH="$LATEST_CHECKPOINT/unet.pth"
fi

python3 src/eval_temporal.py \
    --dataset_path $DATASET_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME/test_results \
    --batch_size 1 \
    --num_workers_test 1 \
    --guidance_scale 7.5 \
    --num_inference_steps 50 \
    --no_pose True 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== DPO Training and Evaluation Completed ===" | tee -a "$LOG_FILE"
echo "Results saved in: $OUTPUT_DIR/$EXPERIMENT_NAME" | tee -a "$LOG_FILE"

# Print summary statistics
echo "" | tee -a "$LOG_FILE"
echo "Training Summary:" | tee -a "$LOG_FILE"
echo "- Training logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "- Model checkpoints: $OUTPUT_DIR/$EXPERIMENT_NAME/checkpoint-*" | tee -a "$LOG_FILE"
echo "- Final model: $OUTPUT_DIR/$EXPERIMENT_NAME/final_model" | tee -a "$LOG_FILE"
echo "- Test results: $OUTPUT_DIR/$EXPERIMENT_NAME/test_results" | tee -a "$LOG_FILE"