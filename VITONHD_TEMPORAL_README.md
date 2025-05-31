# Temporal VITONhd Training Guide

This guide explains how to train the temporal garment prediction model with your VITONhd-format dataset.

## 📁 Dataset Structure

Your dataset is organized as follows:

```
dataset_vitonhd_format/
├── captions.json          # Product captions for all items
├── train/
│   ├── images/
│   │   ├── top5acc/
│   │   │   ├── 2018-week38/
│   │   │   │   ├── 0647982001.jpg
│   │   │   │   └── ...
│   │   │   └── 2020-week7/
│   │   ├── top5gfb/
│   │   ├── top5glb/
│   │   ├── top5gub/
│   │   ├── top5shoe/
│   │   └── top5underwear/
│   ├── im_sketch/         # Same structure as images/
│   └── im_seg/            # Same structure as images/
├── val/                   # Same structure as train/
└── test/                  # Same structure as train/
```

## 🎯 How the System Handles Your Data

### 1. **Six Categories**
The system processes all 6 categories independently:
- `top5acc` - Accessories
- `top5gfb` - Garments Full Body
- `top5glb` - Garments Lower Body
- `top5gub` - Garments Upper Body
- `top5shoe` - Shoes
- `top5underwear` - Underwear

### 2. **Temporal Organization**
- Weeks are sorted chronologically (e.g., 2018-week38 → 2020-week7)
- Each product in week `t` uses past weeks `[t-4, t-3, t-2, t-1]` for conditioning
- Temporal weights decay exponentially (0.8^n for n weeks ago)

### 3. **Input Processing**
- **Images**: Product photos from `images/category/week/`
- **Sketches**: Edge representations from `im_sketch/category/week/`
- **Segmentation**: Garment masks from `im_seg/category/week/`
- **Captions**: Text descriptions from `captions.json`
- **Pose Maps**: Not used (set to zeros as per your requirement)

## 🚀 Quick Start

### 1. **Basic Training**

```bash
# Make script executable
chmod +x train_vitonhd_temporal.sh

# Run training
./train_vitonhd_temporal.sh
```

### 2. **Custom Training with Specific Categories**

```bash
# Train only on specific categories
python3 src/train_temporal.py \
    --dataset_path /root/multimodal-garment-designer/dataset_vitonhd_format \
    --output_dir ./checkpoints \
    --categories top5gub top5glb \  # Only upper and lower body garments
    --num_past_weeks 4 \
    --temporal_weight_decay 0.8 \
    --learning_rate 1e-5 \
    --batch_size 2 \
    --max_train_steps 5000
```

### 3. **Evaluation**

```bash
# Evaluate on test set
python3 src/eval_temporal.py \
    --dataset_path /root/multimodal-garment-designer/dataset_vitonhd_format \
    --checkpoint_path ./checkpoints/final_model/unet.pth \
    --output_dir ./results \
    --categories top5gub \  # Evaluate specific category
    --batch_size 1 \
    --guidance_scale 7.5
```

## ⚙️ Key Parameters

### Temporal Settings
- `--num_past_weeks`: Number of past weeks to consider (default: 4)
- `--temporal_weight_decay`: Weight decay factor for older weeks (default: 0.8)
  - Week t-1: weight = 1.0
  - Week t-2: weight = 0.8
  - Week t-3: weight = 0.64
  - Week t-4: weight = 0.512

### Training Settings
- `--batch_size`: Batch size per GPU (default: 2)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
  - Effective batch size = batch_size × gradient_accumulation_steps
- `--learning_rate`: Learning rate for UNet (default: 1e-5)
- `--temporal_loss_weight`: Weight for temporal consistency loss (default: 0.3)

### Category Filtering
- `--categories`: List of categories to train/evaluate on
  - Example: `--categories top5gub top5glb top5acc`
  - If not specified, uses all 6 categories

## 🧠 How Temporal Weighting Works

For each target week, the system:

1. **Selects Past Weeks**: Takes 4 previous weeks from the same category
2. **Applies Weights**: More recent weeks get higher weights
3. **Aggregates Features**:
   - Sketches: Weighted average of past sketches
   - Captions: Combines past descriptions with temporal context
4. **Generates Prediction**: Uses aggregated conditioning to predict next week

Example caption evolution:
```
Week t-3: "Black lace bra with scalloped trim"
Week t-2: "Modern minimalist tank top"
Week t-1: "Sleek athletic sports bra"
→ Combined: "trending style: Sleek athletic sports bra, evolved from: Black lace bra...; Modern minimalist..."
```

## 📊 Training Tips

### Memory Optimization
```bash
# For limited GPU memory
--batch_size 1 --gradient_accumulation_steps 8
--mixed_precision fp16
```

### Multi-GPU Training
```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch src/train_temporal.py [args...]
```

### Monitoring with Weights & Biases
```bash
# Login to W&B
wandb login

# Training will automatically log to your project
--use_wandb --project_name "temporal-vitonhd"
```

## 🔍 Output Structure

After training and evaluation:

```
temporal_vitonhd_checkpoints/
└── temporal_vitonhd_YYYYMMDD_HHMMSS/
    ├── checkpoint-500/
    │   └── unet.pth
    ├── checkpoint-1000/
    │   └── unet.pth
    ├── final_model/
    │   └── unet.pth
    └── test_results/
        ├── predictions/
        │   ├── top5acc/
        │   │   ├── 0647982001_pred_0000.jpg
        │   │   └── ...
        │   ├── top5gfb/
        │   └── ...
        ├── predictions_metadata.json
        └── category_statistics.json
```

## 🐛 Troubleshooting

### Common Issues

1. **"Not enough weeks for temporal window"**
   - Some categories might not have enough historical data
   - Solution: Use `--num_past_weeks 2` or filter categories

2. **CUDA Out of Memory**
   ```bash
   --batch_size 1 --gradient_accumulation_steps 8
   ```

3. **Missing Sketches/Segmentation**
   - The dataset class generates dummy data if files are missing
   - Check logs for warnings about missing files

4. **Slow Data Loading**
   ```bash
   --num_workers 8  # Increase workers
   ```

## 📈 Expected Results

After training, the model should:
- Predict garments that follow temporal trends within each category
- Generate style-consistent items based on past weeks
- Weight recent trends more heavily than older ones
- Maintain category-specific characteristics

## 🔧 Advanced Usage

### Custom Temporal Weights
Modify in `TemporalVitonHDDataset._create_temporal_samples()`:
```python
# Linear decay instead of exponential
weights = [(len(past_weeks) - j) / len(past_weeks) for j in range(len(past_weeks))]

# Custom decay rate
weights = [0.9 ** (len(past_weeks) - j - 1) for j in range(len(past_weeks))]
```

### Different Loss Weights
```bash
# Emphasize temporal consistency
--temporal_loss_weight 0.5

# Focus on denoising quality
--temporal_loss_weight 0.1
```

### Category-Specific Models
Train separate models for each category:
```bash
for category in top5acc top5gfb top5glb top5gub top5shoe top5underwear; do
    python3 src/train_temporal.py \
        --categories $category \
        --output_dir ./models/$category \
        [other args...]
done
```

## 📚 Next Steps

1. **Experiment with temporal windows**: Try different `num_past_weeks` values
2. **Adjust weighting schemes**: Modify temporal decay rates
3. **Category-specific tuning**: Different categories might benefit from different settings
4. **Ensemble predictions**: Combine multiple models for better results