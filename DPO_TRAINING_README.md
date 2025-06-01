# DPO Training for Temporal Fashion Generation

## Overview

This directory contains the implementation of **Direct Preference Optimization (DPO)** for temporal fashion generation. DPO is a training technique that directly optimizes the model to prefer better outputs over worse ones, making the training more robust and improving output quality.

## What is DPO?

Direct Preference Optimization (DPO) is a method that:
1. **Generates multiple candidates** for each training sample
2. **Scores them using objective metrics** (CLIP-I and CLIP-T scores)
3. **Creates preference pairs** (better vs worse outputs)
4. **Trains the model** to prefer high-quality outputs using a preference loss

## Key Components

### 1. Candidate Generation (`generate_candidates`)
- Generates **20 candidate images** for each training sample
- Uses different random seeds for variety
- Faster inference (20 steps) to reduce computational cost

### 2. CLIP-based Scoring (`CLIPScorer`)
- **CLIP-I Score**: Image-to-image similarity (generated vs target sketch)
- **CLIP-T Score**: Text-to-image similarity (generated vs text prompt)
- **Weighted Score**: Combines both scores (default: 60% CLIP-I, 40% CLIP-T)

### 3. Preference Pair Creation
- Ranks candidates by weighted CLIP scores
- Creates preference pairs (best vs worst, second-best vs second-worst, etc.)
- Up to 10 preference pairs per sample

### 4. DPO Loss
- Computes preference loss between model and reference predictions
- Uses KL divergence regularization (beta parameter)
- Encourages model to generate preferred outputs

## Training Architecture

```
Training Step:
├── 30% chance: Generate candidates + DPO loss
│   ├── Generate 20 candidates
│   ├── Score with CLIP metrics
│   ├── Create preference pairs
│   └── Compute DPO loss
└── 100% chance: Standard diffusion loss + temporal loss

Total Loss = Diffusion Loss + α×Temporal Loss + β×DPO Loss
```

## Key Parameters

### DPO Parameters
- `--num_candidates`: Number of candidates to generate (default: 20)
- `--dpo_beta`: KL regularization strength (default: 0.1)
- `--dpo_weight`: Weight for DPO loss vs diffusion loss (default: 0.5)
- `--clip_i_weight`: Weight for CLIP-I score (default: 0.6)
- `--clip_t_weight`: Weight for CLIP-T score (default: 0.4)

### Training Parameters
- `--learning_rate`: Learning rate (default: 1e-5)
- `--max_train_steps`: Total training steps (default: 1000)
- `--batch_size`: Batch size (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 16)

## Usage

### 1. Basic DPO Training
```bash
python train_vitonhd_dpo.py \
    --dataset_path /path/to/temporal_vitonhd_dataset \
    --output_dir ./dpo_checkpoints \
    --max_train_steps 1500 \
    --num_candidates 20 \
    --dpo_weight 0.5
```

### 2. Using the Helper Script
```bash
python run_dpo_training.py
```

### 3. Fine-tuning from Existing Checkpoint
```bash
python train_vitonhd_dpo.py \
    --dataset_path /path/to/dataset \
    --output_dir ./dpo_checkpoints \
    --resume_from_checkpoint /path/to/checkpoint.pth \
    --learning_rate 5e-6 \
    --dpo_weight 0.7
```

## File Structure

```
├── train_vitonhd_dpo.py      # Main DPO training script
├── run_dpo_training.py       # Helper script with examples
├── DPO_TRAINING_README.md    # This file
└── temporal_vitonhd_dpo_checkpoints/  # Output directory
    └── temporal_vitonhd_dpo_YYYYMMDD_HHMMSS/
        └── checkpoint-XXXX/
            ├── unet.pth
            └── training_config.json
```

## Benefits of DPO Training

### 1. **Quality Improvement**
- Models learn to generate higher-quality images
- Preference optimization guides towards better outputs
- CLIP-based scoring ensures semantic coherence

### 2. **Robust Training**
- Multiple candidates provide diverse training signals
- Preference pairs create strong learning signals
- Temporal consistency maintained alongside quality

### 3. **Objective Metrics**
- CLIP-I ensures visual similarity to target sketches
- CLIP-T ensures text-image alignment
- Weighted combination balances both aspects

## Computational Considerations

### Memory Usage
- Generating 20 candidates increases memory usage
- Consider reducing `num_candidates` if memory limited
- Use `gradient_accumulation_steps` to maintain effective batch size

### Training Time
- DPO adds ~30% overhead (candidates generated 30% of time)
- Faster inference (20 steps) for candidate generation
- Can adjust `dpo_weight` to balance quality vs speed

### GPU Requirements
- Recommended: 24GB+ VRAM for full DPO training
- For 16GB VRAM: reduce `num_candidates` to 10-15
- For 12GB VRAM: reduce to 5-10 candidates

## Monitoring Training

### Key Metrics to Watch
```
Step XXX: diffusion_loss=0.1234, temporal_loss=0.0567, dpo_loss=0.0890
```

- **diffusion_loss**: Standard denoising loss
- **temporal_loss**: Temporal consistency loss
- **dpo_loss**: Preference optimization loss

### Expected Behavior
- DPO loss should decrease over time
- Diffusion loss should remain stable
- Temporal loss should decrease gradually

## Comparison with Standard Training

| Aspect | Standard Training | DPO Training |
|--------|------------------|--------------|
| **Quality** | Good | Better |
| **Robustness** | Standard | Enhanced |
| **Training Time** | Faster | ~30% slower |
| **Memory Usage** | Lower | Higher |
| **Output Consistency** | Variable | More consistent |

## Tips for Best Results

### 1. **Hyperparameter Tuning**
- Start with default parameters
- Increase `dpo_weight` if quality is priority
- Adjust CLIP weights based on your preference (visual vs semantic)

### 2. **Computational Optimization**
- Use mixed precision (`fp16`) to save memory
- Reduce `num_candidates` if training is too slow
- Monitor GPU utilization

### 3. **Quality Assessment**
- Generate samples during training to monitor progress
- Compare with baseline model outputs
- Use both automatic metrics and human evaluation

## Troubleshooting

### Common Issues

**1. Out of Memory**
```bash
# Reduce candidates
--num_candidates 10

# Reduce inference steps
--num_inference_steps 15
```

**2. Slow Training**
```bash
# Reduce DPO frequency (internal parameter)
# Or reduce number of candidates
--num_candidates 15
```

**3. Poor Quality**
```bash
# Increase DPO weight
--dpo_weight 0.7

# Adjust CLIP weights
--clip_i_weight 0.7 --clip_t_weight 0.3
```

## Next Steps

After DPO training:
1. **Evaluate** the model using `eval_temporal.py`
2. **Compare** results with baseline model
3. **Fine-tune** hyperparameters if needed
4. **Deploy** the best checkpoint for inference

## Citation

If you use this DPO implementation, please cite:

```bibtex
@article{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}
```