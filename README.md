# FashionDistill: Generative Design Synthesis via Bestseller Pattern Extraction from the H&M Dataset

## 📝 Overview

Fashion design generation aims to assist designers by providing visual inspiration that reflects diverse trend elements under extremely short product cycles. However, existing models often suffer from limited diversity and rely heavily on supervised learning paradigms, which restrict their adaptability to fast-changing market dynamics.

**FashionDistill** addresses these limitations through **Direct Preference Optimization (DPO)** fine-tuning of fashion generation models. Our framework offers a generalizable strategy that enables better alignment with real-world fashion preferences via multi-modal feedback (text-to-text, image-to-image, image-to-text evaluations).

**Key Contributions:**
- 🎯 **Temporal Trend Modeling**: Extract bestseller patterns from H&M dataset organized by weekly fashion trends
- 🔄 **Direct Preference Optimization**: Fine-tune models using preference pairs based on CLIP-score evaluations
- 🎨 **Multi-Modal Feedback**: Integrate text-to-text, image-to-image, and image-to-text evaluation mechanisms
- 📊 **Real-World Validation**: Experiments on H&M transaction datasets demonstrate effectiveness in reflecting evolving market trends

## 🚀 Quick Start

### Prerequisites
- **CUDA 12.1** (required), **>24GB VRAM** recommended
- **Linux** (tested on Ubuntu)
- **Conda Environment** (required)

### Installation
```bash
# 1. Create environment
conda create -n MGD python=3.9 -y && conda activate MGD
python -m pip install gdown && apt-get update && apt-get install -y unzip

# 2. Install PyTorch (CUDA 12.1)
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
python -m pip install -r requirements.txt
bash apt_get_requirements.sh
```

### Get Dataset and Checkpoint
```bash
bash get_dataset.sh  # Downloads H&M bestseller dataset in VITON-HD format
bash get_ckpt.sh  # Downloads FashionDistill checkpoint
bash get_ref_img.sh  # Downloads reference images for Clip Score calculation
bash get_inference_result_example.sh  # Downloads inference results for metrics calculation
```

### Training
```bash
# DPO training
chmod +x train_vitonhd_dpo.sh && ./train_vitonhd_dpo.sh
```
or write your own training command.
```bash
python3 src/train_vitonhd_dpo.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    $([ ! -z "$RESUME_CHECKPOINT" ] && echo "--resume_from_checkpoint $RESUME_CHECKPOINT") \
    --num_past_weeks 8 \
    --temporal_weight_decay 0.8 \
    --temporal_loss_weight 0.3 \
    --num_candidates 2 \
    --dpo_beta 0.1 \
    --dpo_weight 0.5 \
    --clip_i_weight 0.6 \
    --clip_t_weight 0.4 \
    --dpo_frequency 0.05 \
    --num_inference_steps 10 \
    --learning_rate 5e-6 \
    --max_train_steps 5000 \
    --batch_size 1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 16 \
    --save_steps 500 \
    --num_workers 1 \
    --project_name "temporal-vitonhd-dpo-resumed" \
    --seed 42 \
    --use_wandb 2>&1 | tee -a "$LOG_FILE"
```


### Inference
```bash
chmod +x inference_vitonhd_dpo.sh && ./inference_vitonhd_dpo.sh
```
or write your own inference command.
```bash
python3 src/eval_temporal.py \
    --dataset_path $DATASET_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --output_dir $OUTPUT_DIR/$EXPERIMENT_NAME \
    --num_past_weeks 8 \
    --temporal_weight_decay 0.8 \
    --batch_size 1 \
    --num_workers_test 1 \
    --guidance_scale 5.0 \
    --num_inference_steps 50 \
    --mixed_precision fp16 \
    --seed 42 \
    --no_pose True 2>&1 | tee -a "$LOG_FILE"
```

### Calculate CLIP Score
```bash
chmod +x evaluate_clip_scores.sh && ./evaluate_clip_scores.sh
```
or write your own inference command.
```bash
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

```

## 📂 Dataset Structure

### H&M Integration
We extract bestseller patterns from [H&M dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) weekly, converting to [VITON-HD format](https://github.com/shadow2496/VITON-HD).

The orignal H&M dataset link: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

The VITON-HD format link: https://github.com/shadow2496/VITON-HD

Our preprocessed dataset can be download from this link:
https://drive.google.com/file/d/1C2W0TaHGRpJkrVANWkLkpxDBdng9tqeh/view

```
dataset_vitonhd_format/
├── captions.json
├── train/val/test/
│   ├── images/
│   │   ├── top5acc/    # Accessories
│   │   │   ├── 2018-week38/
│   │   │   └── 2020-week7/
│   │   ├── top5gfb/    # Full Body
│   │   ├── top5glb/    # Lower Body
│   │   ├── top5gub/    # Upper Body
│   │   ├── top5shoe/   # Shoes
│   │   └── top5underwear/
│   ├── im_sketch/      # Edge representations
│   └── im_seg/         # Segmentation masks
```

## ⚗️ Execution Guide

### Training Commands

<details>
<summary><b>🔧 Advanced Training Configuration</b></summary>

#### Temporal Training
```bash
python3 src/train_temporal.py \
    --dataset_path /root/multimodal-garment-designer/dataset_vitonhd_format \
    --output_dir ./temporal_vitonhd_checkpoints \
    --categories top5gub top5glb top5acc \
    --num_past_weeks 4 --temporal_weight_decay 0.8 --temporal_loss_weight 0.3 \
    --learning_rate 1e-5 --max_train_steps 5000 --batch_size 2 \
    --mixed_precision fp16 --save_steps 500
```

#### DPO Training
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/train_vitonhd_dpo.py \
    --dataset_path /root/multimodal-garment-designer/dataset_vitonhd_format \
    --output_dir ./temporal_vitonhd_dpo_checkpoints \
    --num_past_weeks 8 --temporal_weight_decay 0.8 --temporal_loss_weight 0.3 \
    --num_candidates 20 --dpo_beta 0.1 --dpo_weight 0.5 \
    --clip_i_weight 0.6 --clip_t_weight 0.4 --dpo_frequency 0.01 \
    --learning_rate 1e-5 --max_train_steps 5000 --batch_size 1 \
    --mixed_precision fp16 --gradient_accumulation_steps 16
```

#### Memory-Optimized (<24GB VRAM)
```bash
python3 src/train_vitonhd_dpo.py \
    --batch_size 1 --gradient_accumulation_steps 32 --num_candidates 10 \
    --mixed_precision fp16 --num_workers 1 --max_train_steps 3000
```

#### Multi-GPU
```bash
accelerate config  # Run once
accelerate launch src/train_vitonhd_dpo.py [args...]
```
</details>

### Inference Commands

<details>
<summary><b>🔮 Advanced Inference Configuration</b></summary>

#### Temporal Model Evaluation
```bash
python3 src/eval_temporal.py \
    --dataset_path /root/multimodal-garment-designer/dataset_vitonhd_format \
    --checkpoint_path ./temporal_vitonhd_checkpoints/final_model/unet.pth \
    --output_dir ./temporal_results \
    --guidance_scale 7.5 --num_inference_steps 50 --batch_size 1
```

#### DPO Model Evaluation
```bash
python3 src/eval_temporal.py \
    --checkpoint_path ./temporal_vitonhd_dpo_checkpoints/final_model/unet.pth \
    --output_dir ./dpo_results --mixed_precision fp16
```

#### Category-Specific & High-Quality Inference
```bash
# Specific categories
python3 src/eval_temporal.py --categories top5gub top5glb [other-args...]

# High quality (slower)
python3 src/eval_temporal.py --guidance_scale 10.0 --num_inference_steps 100 [other-args...]

# Batch process all categories
for category in top5acc top5gfb top5glb top5gub top5shoe top5underwear; do
    python3 src/eval_temporal.py --categories $category --output_dir ./batch_results/$category [other-args...]
done
```
</details>



### Key Parameters

| **Category** | **Parameter** | **Description** | **Default** |
|--------------|---------------|-----------------|-------------|
| **Temporal** | `--num_past_weeks` | Past weeks for trend analysis | 4 (temporal), 8 (DPO) |
| | `--temporal_weight_decay` | Exponential decay for older weeks | 0.8 |
| **DPO** | `--num_candidates` | Candidates per sample | 20 |
| | `--dpo_weight` | DPO loss weight | 0.5 |
| | `--clip_i_weight / --clip_t_weight` | Image/Text similarity weights | 0.6 / 0.4 |
| **Performance** | `--batch_size` | Training batch size | 1 (DPO), 2 (temporal) |
| | `--mixed_precision` | Memory efficiency | fp16 |
| **Inference** | `--guidance_scale` | Generation guidance | 7.5 |
| | `--num_inference_steps` | Denoising steps | 50 |
| **CLIP Eval** | `--clip_i_weight / --clip_t_weight` | Image/Text evaluation weights | 0.6 / 0.4 |
| | `--batch_size` | Evaluation batch size | 8 |
| | `--create_visualizations` | Generate analysis plots | False |

## 🧠 Technical Architecture

### Model Components
**Backbone**: `runwayml/stable-diffusion-inpainting` + Multimodal Garment Designer UNet

```python
# 13-channel conditioning (9 without pose)
conditioning = torch.cat([
    noisy_latents,        # Denoising target (4)
    masks_resized,        # Inpainting mask (1)
    masked_image_latents, # Masked input (4)
    pose_maps_resized,    # Human pose (3) - optional
    sketches_resized      # Garment sketch (1)
], dim=1)
```

### Loss Functions
```python
total_loss = diffusion_loss + α×temporal_loss + β×dpo_loss
```

**1. Diffusion Loss**: Standard denoising with SNR weighting
```python
snr_weights = compute_snr_weights(timesteps, noise_scheduler, gamma=5.0)
main_loss = (F.mse_loss(model_pred, target, reduction="none") * snr_weights).mean()
```

**2. Temporal Consistency Loss**: Smooth transitions between weeks
```python
# Weighted aggregation: recent weeks have higher influence (0.8^n decay)
weighted_past_features = sum(weight * past_sketch for weight, past_sketch in zip(weights, past_sketches))
temporal_loss = F.mse_loss(current_features, weighted_past_features) * temporal_weight
```

**3. DPO Loss**: Preference optimization via CLIP scoring
```python
# 4-step process: Generate → Score → Pair → Optimize
candidates = generate_candidates(mgd_pipe, batch, num_candidates=20)
scores = score_candidates(candidates, clip_scorer, clip_i_weight=0.6, clip_t_weight=0.4)
dpo_loss = compute_dpo_loss(model_preds, ref_preds, beta=0.1)
```

**DPO Training Strategy**: Applied to 1% of steps after warmup (step >1500), memory-efficient candidate generation without gradients.

### CLIP Score Evaluation
Quantitative assessment using CLIP-I (image similarity) and CLIP-T (text alignment) scores with weighted combination (0.6×CLIP-I + 0.4×CLIP-T).

### Temporal Inference
```python
# Style evolution for future prediction
if next_week_actual_text:
    style_prompt = ensure_garment_consistency(current_style, next_week_text)
else:
    garment_type = classify_garment(current_style)
    style_prompt = f"trending {garment_type} with contemporary design"
```

## 📊 Output Structure
```
temporal_vitonhd_dpo_checkpoints/
└── temporal_vitonhd_dpo_YYYYMMDD_HHMMSS/
    ├── checkpoint-1000/unet.pth
    ├── final_model/unet.pth
    └── dpo_test_results/
        ├── predictions/[categories]/
        ├── captions_used.txt
        └── category_statistics.json

clip_evaluation_results_YYYYMMDD_HHMMSS/
├── summary.json              # Overall statistics and averages
├── individual_scores.json    # Per-image detailed scores
├── scores.csv               # CSV format for analysis
├── clip_analysis.png        # Visualization plots
└── evaluation.log          # Execution log
```

## 🔍 Troubleshooting

| **Issue** | **Solution** |
|-----------|--------------|
| CUDA OOM | `--batch_size 1 --gradient_accumulation_steps 32 --num_candidates 10` |
| Slow Training | `--num_workers 8 --num_candidates 15` |
| Missing Sketches | Dataset generates dummy data (check logs for warnings) |

## 📚 Citation & License

```bibtex
@article{fashiondistill2024,
  title={FashionDistill: Generative Design Synthesis via Bestseller Pattern Extraction from the H&M Dataset},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

**Acknowledgements**: Built upon Multimodal Garment Designer (ICCV 2023), Direct Preference Optimization, H&M Fashion Dataset, and VITON-HD.

**License**: [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - Non-commercial use with attribution.
