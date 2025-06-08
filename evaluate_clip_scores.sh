#!/bin/bash

# CLIP Score Evaluation Script
echo "=== CLIP Score Evaluation ==="

# Clear GPU memory first
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Configuration
INPUT_DIR="./inference_result_example"
REFERENCE_DIR="./ref_image"
CAPTIONS_FILE="./inference_result_example/captions_used.txt"
OUTPUT_DIR="./clip_evaluation_results_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Create log file
LOG_FILE="$OUTPUT_DIR/evaluation.log"

echo "Starting CLIP evaluation with the following configuration:" | tee -a "$LOG_FILE"
echo "Input directory: $INPUT_DIR" | tee -a "$LOG_FILE"
echo "Reference directory: $REFERENCE_DIR" | tee -a "$LOG_FILE"
echo "Captions file: $CAPTIONS_FILE" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify input directories and files exist
echo "Verifying input paths..." | tee -a "$LOG_FILE"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -d "$REFERENCE_DIR" ]; then
    echo "ERROR: Reference directory not found: $REFERENCE_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -f "$CAPTIONS_FILE" ]; then
    echo "ERROR: Captions file not found: $CAPTIONS_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "✅ All input paths verified!" | tee -a "$LOG_FILE"

# Count images in directories
INPUT_IMAGE_COUNT=$(find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
REF_IMAGE_COUNT=$(find "$REFERENCE_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)

echo "📸 Found $INPUT_IMAGE_COUNT images to evaluate" | tee -a "$LOG_FILE"
echo "🎯 Found $REF_IMAGE_COUNT reference images" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Display evaluation parameters
echo "Evaluation Parameters:" | tee -a "$LOG_FILE"
echo "- CLIP-I weight: 0.6 (image-to-image similarity)" | tee -a "$LOG_FILE"
echo "- CLIP-T weight: 0.4 (text-to-image similarity)" | tee -a "$LOG_FILE"
echo "- Batch size: 8 (for memory efficiency)" | tee -a "$LOG_FILE"
echo "- Device: auto (CUDA if available)" | tee -a "$LOG_FILE"
echo "- Visualizations: enabled" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run CLIP evaluation
echo "🚀 Starting CLIP evaluation..." | tee -a "$LOG_FILE"
python3 evaluate_clip_scores.py \
    --input_dir "$INPUT_DIR" \
    --reference_dir "$REFERENCE_DIR" \
    --captions_file "$CAPTIONS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --clip_i_weight 0.6 \
    --clip_t_weight 0.4 \
    --batch_size 8 \
    --device auto \
    --create_visualizations 2>&1 | tee -a "$LOG_FILE"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "✅ CLIP evaluation completed successfully!" | tee -a "$LOG_FILE"
    echo "📊 Results saved in: $OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Display key results if summary exists
    if [ -f "$OUTPUT_DIR/summary.json" ]; then
        echo "📈 Key Results:" | tee -a "$LOG_FILE"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/summary.json', 'r') as f:
        data = json.load(f)
    print(f'  📸 Images processed: {data[\"processed_images\"]}/{data[\"total_images\"]}')
    print(f'  🎯 CLIP-I average: {data.get(\"clip_i_average\", \"N/A\"):.4f}' if 'clip_i_average' in data else '  🎯 CLIP-I: Not computed (no reference images)')
    print(f'  📝 CLIP-T average: {data[\"clip_t_average\"]:.4f}')
    print(f'  🎯 Combined score: {data[\"combined_average\"]:.4f}')
    if data['failed_images'] > 0:
        print(f'  ❌ Failed images: {data[\"failed_images\"]}')
except Exception as e:
    print(f'Could not read summary: {e}')
" | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "📋 Generated Files:" | tee -a "$LOG_FILE"
    echo "  - summary.json: Overall statistics" | tee -a "$LOG_FILE"
    echo "  - individual_scores.json: Per-image detailed scores" | tee -a "$LOG_FILE"
    echo "  - scores.csv: CSV format for analysis" | tee -a "$LOG_FILE"
    echo "  - clip_analysis.png: Visualization plots" | tee -a "$LOG_FILE"
    echo "  - evaluation.log: This log file" | tee -a "$LOG_FILE"

else
    echo "" | tee -a "$LOG_FILE"
    echo "❌ CLIP evaluation failed!" | tee -a "$LOG_FILE"
    echo "Check the log above for error details." | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "=== CLIP Evaluation Completed ===" | tee -a "$LOG_FILE"
echo "Results directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"