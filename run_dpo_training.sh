#!/bin/bash

# ==============================================================================
# DPO Training Script for Temporal Fashion Generation
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DATASET_PATH="/root/multimodal-garment-designer/dataset_vitonhd_format"
OUTPUT_DIR="./temporal_vitonhd_dpo_checkpoints"
PRETRAINED_MODEL="runwayml/stable-diffusion-inpainting"

# Training parameters
LEARNING_RATE="1e-5"
MAX_TRAIN_STEPS="1500"
BATCH_SIZE="1"
GRADIENT_ACCUMULATION_STEPS="16"
MIXED_PRECISION="fp16"
SEED="42"

# DPO specific parameters
NUM_CANDIDATES="20"
DPO_BETA="0.1"
DPO_WEIGHT="0.5"
CLIP_I_WEIGHT="0.6"
CLIP_T_WEIGHT="0.4"

# Temporal parameters
NUM_PAST_WEEKS="4"
TEMPORAL_WEIGHT_DECAY="0.8"
TEMPORAL_LOSS_WEIGHT="0.3"

# Generation parameters
GUIDANCE_SCALE="7.5"
NUM_INFERENCE_STEPS="20"
NO_POSE="True"

# Logging and checkpointing
SAVE_STEPS="250"
LOG_STEPS="50"

# Optional checkpoint to resume from
RESUME_CHECKPOINT=""

# Function to print colored output
print_color() {
    printf "${1}${2}${NC}\n"
}

# Function to print header
print_header() {
    echo ""
    print_color $BLUE "===================================================="
    print_color $BLUE "$1"
    print_color $BLUE "===================================================="
    echo ""
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset PATH          Dataset path (default: $DATASET_PATH)"
    echo "  -o, --output DIR            Output directory (default: $OUTPUT_DIR)"
    echo "  -s, --steps STEPS           Max training steps (default: $MAX_TRAIN_STEPS)"
    echo "  -c, --candidates NUM        Number of candidates (default: $NUM_CANDIDATES)"
    echo "  -w, --dpo-weight WEIGHT     DPO loss weight (default: $DPO_WEIGHT)"
    echo "  -r, --resume PATH           Resume from checkpoint"
    echo "  --learning-rate LR          Learning rate (default: $LEARNING_RATE)"
    echo "  --batch-size SIZE           Batch size (default: $BATCH_SIZE)"
    echo "  --clip-i-weight WEIGHT      CLIP-I weight (default: $CLIP_I_WEIGHT)"
    echo "  --clip-t-weight WEIGHT      CLIP-T weight (default: $CLIP_T_WEIGHT)"
    echo "  --fast                      Fast training (fewer candidates, steps)"
    echo "  --gpu-check                 Check GPU before training"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run with default parameters"
    echo "  $0 --fast                   # Run with faster settings"
    echo "  $0 -s 1000 -c 15           # Custom steps and candidates"
    echo "  $0 -r /path/to/checkpoint   # Resume from checkpoint"
}

# Function to check GPU
check_gpu() {
    print_header "GPU Check"

    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        echo ""

        # Check available memory
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        if [ "$FREE_MEM" -lt 12000 ]; then
            print_color $YELLOW "Warning: Low GPU memory ($FREE_MEM MB). Consider reducing num_candidates."
        else
            print_color $GREEN "GPU memory looks good ($FREE_MEM MB available)."
        fi
    else
        print_color $RED "nvidia-smi not found. Cannot check GPU status."
    fi
    echo ""
}

# Function to check if paths exist
check_paths() {
    print_header "Path Verification"

    if [ ! -d "$DATASET_PATH" ]; then
        print_color $RED "Error: Dataset path does not exist: $DATASET_PATH"
        exit 1
    else
        print_color $GREEN "Dataset path verified: $DATASET_PATH"
    fi

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    print_color $GREEN "Output directory ready: $OUTPUT_DIR"

    if [ -n "$RESUME_CHECKPOINT" ] && [ ! -f "$RESUME_CHECKPOINT" ]; then
        print_color $RED "Error: Checkpoint file does not exist: $RESUME_CHECKPOINT"
        exit 1
    elif [ -n "$RESUME_CHECKPOINT" ]; then
        print_color $GREEN "Checkpoint file verified: $RESUME_CHECKPOINT"
    fi
    echo ""
}

# Function to set fast training parameters
set_fast_mode() {
    print_color $YELLOW "Setting fast training mode..."
    NUM_CANDIDATES="10"
    MAX_TRAIN_STEPS="500"
    NUM_INFERENCE_STEPS="15"
    SAVE_STEPS="100"
    LOG_STEPS="25"
    print_color $GREEN "Fast mode configured."
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--steps)
            MAX_TRAIN_STEPS="$2"
            shift 2
            ;;
        -c|--candidates)
            NUM_CANDIDATES="$2"
            shift 2
            ;;
        -w|--dpo-weight)
            DPO_WEIGHT="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --clip-i-weight)
            CLIP_I_WEIGHT="$2"
            shift 2
            ;;
        --clip-t-weight)
            CLIP_T_WEIGHT="$2"
            shift 2
            ;;
        --fast)
            set_fast_mode
            shift
            ;;
        --gpu-check)
            check_gpu
            exit 0
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_header "🎨 Temporal Fashion Generation - DPO Training"

# Show configuration
print_color $BLUE "Training Configuration:"
echo "  Dataset Path: $DATASET_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Max Training Steps: $MAX_TRAIN_STEPS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo ""
print_color $BLUE "DPO Configuration:"
echo "  Number of Candidates: $NUM_CANDIDATES"
echo "  DPO Beta: $DPO_BETA"
echo "  DPO Weight: $DPO_WEIGHT"
echo "  CLIP-I Weight: $CLIP_I_WEIGHT"
echo "  CLIP-T Weight: $CLIP_T_WEIGHT"
echo ""
print_color $BLUE "Temporal Configuration:"
echo "  Past Weeks: $NUM_PAST_WEEKS"
echo "  Temporal Weight Decay: $TEMPORAL_WEIGHT_DECAY"
echo "  Temporal Loss Weight: $TEMPORAL_LOSS_WEIGHT"
echo ""

if [ -n "$RESUME_CHECKPOINT" ]; then
    print_color $YELLOW "Resuming from checkpoint: $RESUME_CHECKPOINT"
    echo ""
fi

# Verify paths and GPU
check_paths
check_gpu

# Build the training command
CMD="python train_vitonhd_dpo.py"
CMD="$CMD --dataset_path $DATASET_PATH"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --pretrained_model_name_or_path $PRETRAINED_MODEL"

# Training parameters
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --max_train_steps $MAX_TRAIN_STEPS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --mixed_precision $MIXED_PRECISION"
CMD="$CMD --seed $SEED"

# DPO parameters
CMD="$CMD --num_candidates $NUM_CANDIDATES"
CMD="$CMD --dpo_beta $DPO_BETA"
CMD="$CMD --dpo_weight $DPO_WEIGHT"
CMD="$CMD --clip_i_weight $CLIP_I_WEIGHT"
CMD="$CMD --clip_t_weight $CLIP_T_WEIGHT"

# Temporal parameters
CMD="$CMD --num_past_weeks $NUM_PAST_WEEKS"
CMD="$CMD --temporal_weight_decay $TEMPORAL_WEIGHT_DECAY"
CMD="$CMD --temporal_loss_weight $TEMPORAL_LOSS_WEIGHT"

# Generation parameters
CMD="$CMD --guidance_scale $GUIDANCE_SCALE"
CMD="$CMD --num_inference_steps $NUM_INFERENCE_STEPS"
CMD="$CMD --no_pose $NO_POSE"

# Logging and checkpointing
CMD="$CMD --save_steps $SAVE_STEPS"
CMD="$CMD --log_steps $LOG_STEPS"

# Add resume checkpoint if specified
if [ -n "$RESUME_CHECKPOINT" ]; then
    CMD="$CMD --resume_from_checkpoint $RESUME_CHECKPOINT"
fi

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/training_log_$TIMESTAMP.txt"

print_header "🚀 Starting DPO Training"
print_color $GREEN "Command: $CMD"
print_color $BLUE "Log file: $LOG_FILE"
echo ""

# Automatically proceed with training (no user confirmation needed)
print_color $GREEN "✅ Proceeding with training automatically..."

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Run the training command
print_color $GREEN "Training started at $(date)"
echo "Command: $CMD" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
    print_color $GREEN "✅ Training completed successfully at $(date)"
    echo "Completed at: $(date)" >> "$LOG_FILE"

    print_header "🎉 Training Summary"
    echo "Log file: $LOG_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    print_color $GREEN "Next steps:"
    echo "1. Check the output directory for saved checkpoints"
    echo "2. Use the trained model for evaluation with eval_temporal.py"
    echo "3. Compare results with baseline model"
else
    print_color $RED "❌ Training failed. Check the log file for details."
    echo "Failed at: $(date)" >> "$LOG_FILE"
    exit 1
fi