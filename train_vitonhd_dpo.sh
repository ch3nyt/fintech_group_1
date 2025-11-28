#!/usr/bin/env bash
set -euo pipefail

# --- venv 與 repo 路徑 ---
VENV="/content/env"
REPO="/content/fashiondistill"

# 讓 python 指向 venv；並把 src 版型加進匯入路徑
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$REPO/src:$REPO"
export WANDB_MODE=disabled

cd "$REPO"

echo "=== DPO-Enhanced Temporal VITONhd Training ==="
echo "Clearing GPU memory..."
python - <<'PY'
import torch
torch.cuda.empty_cache()
print("CUDA available:", torch.cuda.is_available())
PY

# ---- 基本設定 ----
DATASET_PATH="./dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_dpo_checkpoints"
EXPERIMENT_NAME="temporal_vitonhd_dpo_$(date +%Y%m%d_%H%M%S)"
RESUME_CHECKPOINT=""

mkdir -p "$OUTPUT_DIR/$EXPERIMENT_NAME"
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/training.log"

echo "Starting DPO training with the following configuration:" | tee -a "$LOG_FILE"
echo "Dataset: $DATASET_PATH" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Experiment: $EXPERIMENT_NAME" | tee -a "$LOG_FILE"
echo "Resume from: $RESUME_CHECKPOINT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "DPO Parameters (Memory Optimized):" | tee -a "$LOG_FILE"
echo "- Number of candidates: 2" | tee -a "$LOG_FILE"
echo "- DPO beta: 0.1" | tee -a "$LOG_FILE"
echo "- DPO weight: 0.5" | tee -a "$LOG_FILE"
echo "- CLIP-I/T weights: 0.6 / 0.4" | tee -a "$LOG_FILE"
echo "- DPO frequency: 0.01" | tee -a "$LOG_FILE"
echo "- Inference steps: 10" | tee -a "$LOG_FILE"
echo "- LR: 5e-6" | tee -a "$LOG_FILE"
echo "- Max steps: 50" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [[ -n "$RESUME_CHECKPOINT" && ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found: $RESUME_CHECKPOINT" | tee -a "$LOG_FILE"
  exit 1
fi
[[ -n "$RESUME_CHECKPOINT" ]] && echo "Checkpoint verified: $RESUME_CHECKPOINT" | tee -a "$LOG_FILE" || echo "No checkpoint provided - starting fresh training" | tee -a "$LOG_FILE"

echo "Starting DPO training..." | tee -a "$LOG_FILE"

# 重要：每行最後的 '\' 後面**不可**有空白
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u -m src.train_vitonhd_dpo \
  --pretrained_model_name_or_path "alwold/stable-diffusion-2-inpainting" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME" \
  $([[ -n "$RESUME_CHECKPOINT" ]] && echo --resume_from_checkpoint "$RESUME_CHECKPOINT") \
  --num_past_weeks 8 \
  --temporal_weight_decay 0.8 \
  --temporal_loss_weight 0.3 \
  --num_candidates 2 \
  --dpo_beta 0.1 \
  --dpo_weight 0.5 \
  --clip_i_weight 0.6 \
  --clip_t_weight 0.4 \
  --dpo_frequency 0.1 \
  --num_inference_steps 10 \
  --learning_rate 5e-6 \
  --max_train_steps 500 \
  --batch_size 1 \
  --mixed_precision fp16 \
  --gradient_accumulation_steps 16 \
  --save_steps 50 \
  --num_workers 1 \
  --project_name "temporal-vitonhd-dpo-resumed" \
  --seed 42 \
  --use_wandb 2>&1 | tee -a "$LOG_FILE"

echo "DPO Training completed!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Running evaluation on test set with DPO-trained model..." | tee -a "$LOG_FILE"

LATEST_CHECKPOINT="$(find "$OUTPUT_DIR/$EXPERIMENT_NAME" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n1 || true)"
if [[ -z "$LATEST_CHECKPOINT" ]]; then
  echo "No checkpoint found, using final model..." | tee -a "$LOG_FILE"
  CHECKPOINT_PATH="$OUTPUT_DIR/$EXPERIMENT_NAME/final_model/unet.pth"
else
  echo "Using checkpoint: $LATEST_CHECKPOINT" | tee -a "$LOG_FILE"
  CHECKPOINT_PATH="$LATEST_CHECKPOINT/unet.pth"
fi

python -u -m src.eval_temporal \
  --dataset_path "$DATASET_PATH" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME/test_results" \
  --batch_size 1 \
  --num_workers_test 1 \
  --guidance_scale 7.5 \
  --num_inference_steps 50 \
  --no_pose True 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== DPO Training and Evaluation Completed ===" | tee -a "$LOG_FILE"
echo "Results saved in: $OUTPUT_DIR/$EXPERIMENT_NAME" | tee -a "$LOG_FILE"
