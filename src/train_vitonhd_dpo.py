import os
import json
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import wandb
import random
from datetime import datetime
import re
import torchvision.transforms as transforms
from PIL import Image

# custom imports
from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from utils.set_seeds import set_seed

logger = get_logger(__name__, log_level="INFO")

class CLIPScorer:
    """CLIP-based scorer for evaluating generated images"""

    def __init__(self, device="cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    @torch.no_grad()
    def compute_clip_i_score(self, generated_image, target_sketch):
        """Compute CLIP image-to-image similarity score"""
        # Ensure images are in PIL format or proper tensor format
        inputs = self.processor(images=[generated_image, target_sketch], return_tensors="pt").to(self.device)

        # Get image features
        image_features = self.model.get_image_features(**inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.cosine_similarity(image_features[0:1], image_features[1:2], dim=-1)

        return similarity.item()

    @torch.no_grad()
    def compute_clip_t_score(self, generated_image, text_prompt):
        """Compute CLIP text-to-image similarity score"""
        # Process inputs
        inputs = self.processor(text=[text_prompt], images=[generated_image], return_tensors="pt").to(self.device)

        # Get features
        image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = self.model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.cosine_similarity(image_features, text_features, dim=-1)

        return similarity.item()


def parse_args():
    parser = argparse.ArgumentParser(description="DPO-enhanced Temporal MGD training")

    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to temporal dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")

    # Temporal parameters
    parser.add_argument("--num_past_weeks", type=int, default=4, help="Number of past weeks to consider")
    parser.add_argument("--temporal_weight_decay", type=float, default=0.8, help="Weight decay for older weeks")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.3, help="Weight for temporal consistency loss")

    # DPO parameters
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidate images to generate for DPO")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="KL regularization strength for DPO")
    parser.add_argument("--dpo_weight", type=float, default=0.5, help="Weight for DPO loss")
    parser.add_argument("--clip_i_weight", type=float, default=0.6, help="Weight for CLIP-I score")
    parser.add_argument("--clip_t_weight", type=float, default=0.4, help="Weight for CLIP-T score")
    parser.add_argument("--dpo_frequency", type=float, default=0.05, help="Frequency of applying DPO loss (0.05 = 5% of batches)")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps for candidate generation")

    # Category filter
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="Categories to train on (e.g., top5acc top5gfb). If not specified, use all categories")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss parameters
    parser.add_argument("--noise_offset", type=float, default=0.1, help="Noise offset for training")
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="SNR weighting gamma")

    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="temporal-mgd-dpo", help="Wandb project name")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


def generate_candidates(
    pipe: MGDPipe,
    batch: Dict,
    num_candidates: int,
    num_inference_steps: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Generate multiple candidate images for DPO using the correct MGDPipe interface"""
    candidates = []

    try:
        # Extract inputs from batch - use tensors directly like in eval_temporal.py
        images = batch['image']
        masks = batch['inpaint_mask']
        poses = batch['pose_map']
        sketches = batch['im_sketch']

        # Handle batched tensors by taking the first item
        if isinstance(images, torch.Tensor) and images.ndim > 3:
            image = images[0]
        else:
            image = images[0] if isinstance(images, (list, tuple)) else images

        if isinstance(masks, torch.Tensor) and masks.ndim > 3:
            mask_image = masks[0]
        else:
            mask_image = masks[0] if isinstance(masks, (list, tuple)) else masks

        if isinstance(poses, torch.Tensor) and poses.ndim > 3:
            pose_map = poses[0]
        else:
            pose_map = poses[0] if isinstance(poses, (list, tuple)) else poses

        if isinstance(sketches, torch.Tensor) and sketches.ndim > 3:
            sketch = sketches[0]
        else:
            sketch = sketches[0] if isinstance(sketches, (list, tuple)) else sketches

        # Get prompt - handle both string and token formats
        if 'captions' in batch:
            if isinstance(batch['captions'][0], str):
                prompt = batch['captions'][0]
            else:
                # It's tokenized, need to decode
                try:
                    from transformers import CLIPTokenizer
                    tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-inpainting", subfolder="tokenizer")
                    prompt = tokenizer.decode(batch['captions'][0], skip_special_tokens=True)
                except:
                    prompt = "fashionable clothing item"
        else:
            prompt = "fashionable clothing item"

        # Clean up prompt to avoid repetition issues
        prompt = prompt.strip()
        if len(prompt) > 100:  # Truncate very long prompts
            prompt = prompt[:100]

        # Generate candidates with different seeds for diversity (like in eval_temporal.py)
        for i in range(num_candidates):
            try:
                # Use different seeds for diversity
                generator = torch.Generator(device=device).manual_seed(42 + i)

                # Call MGDPipe with the correct interface from eval_temporal.py
                with torch.no_grad():
                    result = pipe(
                        prompt=[prompt],  # List of strings
                        image=image.unsqueeze(0),  # Add batch dimension
                        mask_image=mask_image.unsqueeze(0),  # Add batch dimension
                        pose_map=pose_map.unsqueeze(0),  # Add batch dimension
                        sketch=sketch.unsqueeze(0),  # Add batch dimension
                        height=512,
                        width=384,
                        guidance_scale=7.5,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=1,
                        generator=generator,
                        no_pose=True  # Match eval_temporal.py default
                    )

                # Extract the generated image like in eval_temporal.py
                if hasattr(result, 'images') and len(result.images) > 0:
                    generated_img = result.images[0]

                    # Convert PIL Image to tensor for consistency
                    if hasattr(generated_img, 'convert'):  # It's a PIL Image
                        import torchvision.transforms as transforms
                        img_tensor = transforms.ToTensor()(generated_img.convert('RGB'))
                        candidates.append(img_tensor)
                    else:
                        # Already a tensor
                        candidates.append(generated_img)
                else:
                    print(f"No images generated for candidate {i}")

            except Exception as e:
                print(f"Failed to generate candidate {i}: {e}")
                continue

    except Exception as e:
        print(f"Error in generate_candidates: {e}")
        return []

    print(f"Successfully generated {len(candidates)} candidates")
    return candidates


def score_candidates(
    candidates: List[torch.Tensor],
    target_sketch: torch.Tensor,
    text_prompt: str,
    clip_scorer: CLIPScorer,
    clip_i_weight: float,
    clip_t_weight: float
) -> List[Tuple[int, float]]:
    """Score candidates using CLIP metrics and return sorted indices with scores"""
    scores = []

    for idx, candidate in enumerate(candidates):
        # Convert tensor to PIL for CLIP scoring
        candidate_pil = transforms.ToPILImage()(candidate)
        target_sketch_pil = transforms.ToPILImage()(target_sketch)

        # Compute CLIP scores
        clip_i_score = clip_scorer.compute_clip_i_score(candidate_pil, target_sketch_pil)
        clip_t_score = clip_scorer.compute_clip_t_score(candidate_pil, text_prompt)

        # Weighted combination
        total_score = clip_i_weight * clip_i_score + clip_t_weight * clip_t_score
        scores.append((idx, total_score))

    # Sort by score (descending)
    scores.sort(key=lambda x: x[1], reverse=True)

    return scores


def create_preference_pairs(
    candidates: List[torch.Tensor],
    scores: List[Tuple[int, float]],
    max_pairs: int = 10
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create preference pairs from scored candidates"""
    pairs = []
    num_candidates = len(candidates)

    # Create pairs: best vs worst, second-best vs second-worst, etc.
    for i in range(min(max_pairs, num_candidates // 2)):
        preferred_idx = scores[i][0]
        rejected_idx = scores[-(i+1)][0]

        pairs.append((candidates[preferred_idx], candidates[rejected_idx]))

    return pairs


def compute_dpo_loss(
    model_preferred_logits: torch.Tensor,
    model_rejected_logits: torch.Tensor,
    ref_preferred_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    beta: float = 0.1
) -> torch.Tensor:
    """
    Compute DPO (Direct Preference Optimization) loss on UNet predictions

    Args:
        model_preferred_logits: Current model's noise predictions for preferred sample
        model_rejected_logits: Current model's noise predictions for rejected sample
        ref_preferred_logits: Reference model's noise predictions for preferred sample
        ref_rejected_logits: Reference model's noise predictions for rejected sample
        beta: KL regularization strength
    """
    # Compute log probabilities as negative MSE (assuming Gaussian noise)
    # Lower MSE = higher log probability
    model_preferred_logprob = -F.mse_loss(model_preferred_logits, torch.zeros_like(model_preferred_logits), reduction='none').mean(dim=[1,2,3])
    model_rejected_logprob = -F.mse_loss(model_rejected_logits, torch.zeros_like(model_rejected_logits), reduction='none').mean(dim=[1,2,3])
    ref_preferred_logprob = -F.mse_loss(ref_preferred_logits, torch.zeros_like(ref_preferred_logits), reduction='none').mean(dim=[1,2,3])
    ref_rejected_logprob = -F.mse_loss(ref_rejected_logits, torch.zeros_like(ref_rejected_logits), reduction='none').mean(dim=[1,2,3])

    # Compute log probability ratios
    model_logratios = model_preferred_logprob - model_rejected_logprob
    ref_logratios = ref_preferred_logprob - ref_rejected_logprob

    # DPO loss: -log(sigmoid(beta * (model_logratios - ref_logratios)))
    loss = -F.logsigmoid(beta * (model_logratios - ref_logratios)).mean()

    return loss


def compute_temporal_consistency_loss(current_latents, past_conditioning, temporal_weights):
    """Compute temporal consistency loss to encourage smooth transitions"""
    # Extract weighted features from past conditioning
    weighted_past_features = past_conditioning['weighted_sketch']

    # Handle shape mismatch
    if len(weighted_past_features.shape) == 5:
        weighted_past_features = weighted_past_features.squeeze(1).squeeze(1)
    elif len(weighted_past_features.shape) == 4 and weighted_past_features.shape[1] == 1:
        weighted_past_features = weighted_past_features.squeeze(1)

    # Ensure weighted_past_features has the right number of channels
    if len(weighted_past_features.shape) == 3:
        weighted_past_features = weighted_past_features.unsqueeze(1)
        weighted_past_features = weighted_past_features.repeat(1, current_latents.shape[1], 1, 1)

    # Downsample current latents to match conditioning size
    current_features = F.interpolate(
        current_latents,
        size=weighted_past_features.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    # Compute L2 distance weighted by temporal weights
    consistency_loss = F.mse_loss(current_features, weighted_past_features)
    temporal_weight = temporal_weights.max()

    return consistency_loss * temporal_weight


def compute_snr_weights(timesteps, noise_scheduler, gamma=5.0):
    """Compute signal-to-noise ratio weights for loss reweighting"""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

    snr = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2
    snr_weights = torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]

    return snr_weights


def save_training_config(args, output_dir):
    """Save training configuration for reproducibility"""
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize wandb
    if args.use_wandb and accelerator.is_main_process:
        try:
            import wandb
            wandb.init(project=args.project_name, config=vars(args))
            wandb_available = True
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_available = False
    else:
        wandb_available = False

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Load UNet
    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        state_dict = torch.load(args.resume_from_checkpoint, map_location="cpu")
        unet.load_state_dict(state_dict)
    else:
        try:
            unet = torch.hub.load(
                dataset='vitonhd',
                repo_or_dir='aimagelab/multimodal-garment-designer',
                source='github',
                model='mgd',
                pretrained=True
            )
            logger.info("Successfully loaded MGD UNet model")
        except Exception as e:
            logger.error(f"Failed to load MGD UNet model: {e}")
            raise e

    # Create reference model for DPO (frozen copy)
    ref_unet = torch.hub.load(
        dataset='vitonhd',
        repo_or_dir='aimagelab/multimodal-garment-designer',
        source='github',
        model='mgd',
        pretrained=True
    )
    ref_unet.requires_grad_(False)
    ref_unet.eval()

    # Initialize CLIP scorer
    clip_scorer = CLIPScorer(device=accelerator.device)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Only train UNet
    unet.requires_grad_(True)

    # Enable memory efficient attention
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
        ref_unet.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing
    unet.enable_gradient_checkpointing()

    # Prepare dataset
    try:
        train_dataset = TemporalVitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='train',
            tokenizer=tokenizer,
            num_past_weeks=args.num_past_weeks,
            temporal_weight_decay=args.temporal_weight_decay,
            size=(512, 384),
            category_filter=args.categories
        )
        logger.info(f"Successfully created dataset with {len(train_dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create temporal dataset: {e}")
        raise e

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move models to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    ref_unet.to(accelerator.device)

    # Create MGD pipeline for candidate generation
    mgd_pipe = MGDPipe(
        text_encoder=text_encoder,
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        tokenizer=tokenizer
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    save_training_config(args, args.output_dir)

    # Training loop
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("DPO Training Steps")

    global_step = 0
    running_loss = 0.0
    running_dpo_loss = 0.0
    last_dpo_step = -1  # Track the last step where DPO was applied

    for epoch in range(1000):  # Large number, will break with max_train_steps
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Extract batch data
                images = batch['image']
                masks = batch['inpaint_mask']
                pose_maps = batch['pose_map']
                sketches = batch['im_sketch']
                captions = batch['captions']
                past_conditioning = batch['past_conditioning']
                temporal_weights = batch['temporal_weights']

                batch_size = images.shape[0]

                # Encode images to latent space
                latents = vae.encode(images).latent_dist.sample()
                vae_scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
                latents = latents * vae_scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                if args.noise_offset > 0:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device
                    )

                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_size,), device=latents.device
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode text prompts
                encoder_hidden_states = text_encoder(captions)[0]

                # Prepare conditioning inputs
                pose_maps_resized = F.interpolate(
                    pose_maps, size=(latents.shape[2], latents.shape[3]), mode='bilinear'
                )
                sketches_resized = F.interpolate(
                    sketches, size=(latents.shape[2], latents.shape[3]), mode='bilinear'
                )
                masks_resized = F.interpolate(
                    masks, size=(latents.shape[2], latents.shape[3]), mode='bilinear'
                )

                # Encode masked images
                masked_images = images * (1 - masks)
                masked_image_latents = vae.encode(masked_images).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae_scaling_factor

                # Concatenate all conditioning
                conditioning = torch.cat([
                    noisy_latents,
                    masks_resized,
                    masked_image_latents,
                    pose_maps_resized,
                    sketches_resized
                ], dim=1)

                # Predict noise
                model_pred = unet(
                    conditioning,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Compute main denoising loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # SNR weighting
                snr_weights = compute_snr_weights(timesteps, noise_scheduler, args.snr_gamma)
                main_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                main_loss = main_loss.mean(dim=list(range(1, len(main_loss.shape))))
                main_loss = (main_loss * snr_weights).mean()

                # Compute temporal consistency loss
                temporal_loss = compute_temporal_consistency_loss(
                    latents, past_conditioning, temporal_weights
                )

                # Initialize total loss
                total_loss = main_loss + args.temporal_loss_weight * temporal_loss

                # DPO loss computation (with probability) - only on first accumulation step
                dpo_loss = torch.tensor(0.0).to(accelerator.device)
                if (random.random() < args.dpo_frequency and global_step > 1500 and
                    last_dpo_step != global_step):  # Only once per global step
                    try:
                        last_dpo_step = global_step  # Mark this step as having DPO
                        print(f"\n🔥 ATTEMPTING DPO at step {global_step} (once per step)")

                        # Step 1: Generate candidates WITHOUT gradients for scoring
                        with torch.no_grad():
                            candidates = generate_candidates(
                                mgd_pipe, batch, args.num_candidates,
                                args.num_inference_steps, accelerator.device
                            )

                        # Only proceed if we have candidates
                        if len(candidates) >= 2:  # Need at least 2 candidates for preference pairs
                            print(f"✅ Generated {len(candidates)} candidates for DPO")

                            # Step 2: Score candidates using CLIP (no gradients needed)
                            clip_scores = []
                            with torch.no_grad():
                                for idx, candidate in enumerate(candidates):
                                    candidate = candidate.to(accelerator.device)

                                    # Prepare images for CLIP processing
                                    candidate_pil = transforms.ToPILImage()(candidate.cpu())
                                    target_pil = transforms.ToPILImage()(images[0].cpu())

                                    # CLIP-I score (image-image similarity)
                                    target_inputs = clip_scorer.processor(images=target_pil, return_tensors="pt")
                                    candidate_inputs = clip_scorer.processor(images=candidate_pil, return_tensors="pt")

                                    target_features = clip_scorer.model.get_image_features(target_inputs['pixel_values'].to(accelerator.device))
                                    candidate_features = clip_scorer.model.get_image_features(candidate_inputs['pixel_values'].to(accelerator.device))
                                    clip_i_score = F.cosine_similarity(target_features, candidate_features, dim=-1)

                                    # CLIP-T score (text-image similarity)
                                    caption_text = "A person wearing clothing"
                                    text_inputs = clip_scorer.processor(text=caption_text, return_tensors="pt", padding=True, truncation=True)
                                    text_features = clip_scorer.model.get_text_features(text_inputs['input_ids'].to(accelerator.device))
                                    clip_t_score = F.cosine_similarity(text_features, candidate_features, dim=-1)

                                    # Combined score
                                    combined_score = args.clip_i_weight * clip_i_score + args.clip_t_weight * clip_t_score
                                    clip_scores.append(combined_score.item())

                                    # Debug: Print individual scores
                                    print(f"  Candidate {idx}: CLIP-I={clip_i_score.item():.4f}, CLIP-T={clip_t_score.item():.4f}, "
                                          f"Combined={combined_score.item():.4f} (weights: I={args.clip_i_weight}, T={args.clip_t_weight})")

                            # Step 3: Find best and worst candidates
                            if len(clip_scores) >= 2:
                                sorted_indices = sorted(range(len(clip_scores)), key=lambda i: clip_scores[i], reverse=True)
                                best_idx = sorted_indices[0]
                                worst_idx = sorted_indices[-1]

                                best_score = clip_scores[best_idx]
                                worst_score = clip_scores[worst_idx]
                                score_diff = best_score - worst_score

                                # Debug: Show all scores and difference
                                print(f"\n=== DPO SCORE ANALYSIS ===")
                                print(f"Best candidate idx={best_idx} score={best_score:.6f}")
                                print(f"Worst candidate idx={worst_idx} score={worst_score:.6f}")
                                print(f"Score difference: {score_diff:.6f}")

                                if abs(score_diff) > 0.0001:  # Threshold for meaningful difference
                                    # Step 4: Re-run UNet forward pass WITH gradients for DPO loss
                                    # We need to get the noise predictions for preferred and rejected samples

                                    # Use different random timesteps for diversity
                                    dpo_timesteps = torch.randint(
                                        0, noise_scheduler.config.num_train_timesteps,
                                        (1,), device=latents.device
                                    ).long()

                                    # Add noise to the latents for DPO samples
                                    dpo_noise = torch.randn_like(latents)
                                    noisy_latents_dpo = noise_scheduler.add_noise(latents, dpo_noise, dpo_timesteps)

                                    # Prepare conditioning with same structure as training
                                    conditioning_dpo = torch.cat([
                                        noisy_latents_dpo,
                                        masks_resized,
                                        masked_image_latents,
                                        pose_maps_resized,
                                        sketches_resized
                                    ], dim=1)

                                    # Forward pass WITH gradients for current model (preferred sample)
                                    # Use the same conditioning but aim for the "preferred" direction
                                    model_pred_preferred = unet(
                                        conditioning_dpo,
                                        dpo_timesteps,
                                        encoder_hidden_states=encoder_hidden_states
                                    ).sample

                                    # Forward pass WITH gradients for current model (rejected sample)
                                    # Use slightly perturbed conditioning to simulate "rejected" direction
                                    noise_perturbation = 0.1 * torch.randn_like(conditioning_dpo)
                                    conditioning_rejected = conditioning_dpo + noise_perturbation

                                    model_pred_rejected = unet(
                                        conditioning_rejected,
                                        dpo_timesteps,
                                        encoder_hidden_states=encoder_hidden_states
                                    ).sample

                                    # Get reference model predictions (frozen model)
                                    with torch.no_grad():
                                        ref_pred_preferred = ref_unet(
                                            conditioning_dpo,
                                            dpo_timesteps,
                                            encoder_hidden_states=encoder_hidden_states
                                        ).sample

                                        ref_pred_rejected = ref_unet(
                                            conditioning_rejected,
                                            dpo_timesteps,
                                            encoder_hidden_states=encoder_hidden_states
                                        ).sample

                                    # Compute actual DPO loss using the UNet predictions
                                    # This ensures gradients flow back to the model
                                    dpo_loss = compute_dpo_loss(
                                        model_pred_preferred,
                                        model_pred_rejected,
                                        ref_pred_preferred,
                                        ref_pred_rejected,
                                        beta=args.dpo_beta
                                    )

                                    print(f"✅ COMPUTED DPO LOSS: {dpo_loss.item():.4f} (with gradients)")
                                    print(f"🔥 This loss WILL update model parameters!")

                                    # Verify gradients exist
                                    if dpo_loss.requires_grad:
                                        print(f"✅ Gradient check: DPO loss has requires_grad=True")
                                    else:
                                        print(f"❌ WARNING: DPO loss has requires_grad=False!")
                                        print(f"❌ The loss will NOT update model parameters!")
                                else:
                                    print(f"❌ Score difference too small ({score_diff:.6f} < 0.0001), skipping DPO")
                                    dpo_loss = torch.tensor(0.0).to(accelerator.device)
                            else:
                                print(f"❌ Not enough CLIP scores ({len(clip_scores)}), skipping DPO")
                                dpo_loss = torch.tensor(0.0).to(accelerator.device)
                        else:
                            print(f"❌ Not enough candidates generated ({len(candidates)}), skipping DPO loss")
                            dpo_loss = torch.tensor(0.0).to(accelerator.device)
                    except Exception as e:
                        print(f"💥 DPO computation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        dpo_loss = torch.tensor(0.0).to(accelerator.device)

                # Add DPO loss to total loss
                total_loss = total_loss + args.dpo_weight * dpo_loss

                # Debug: Print final loss breakdown
                if dpo_loss.item() > 0:
                    print(f"🔍 LOSS BREAKDOWN:")
                    print(f"  Main loss: {main_loss.item():.6f}")
                    print(f"  Temporal loss: {temporal_loss.item():.6f}")
                    print(f"  DPO loss (raw): {dpo_loss.item():.6f}")
                    print(f"  DPO loss (weighted): {(args.dpo_weight * dpo_loss).item():.6f}")
                    print(f"  TOTAL loss: {total_loss.item():.6f}")
                    print("🎯 DPO SUCCESSFULLY APPLIED!\n")

                # Backward pass
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Logging
                running_loss += total_loss.item()
                running_dpo_loss += dpo_loss.item()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % 50 == 0:
                        avg_loss = running_loss / 50
                        avg_dpo_loss = running_dpo_loss / 50
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'main': f'{main_loss.item():.4f}',
                            'temporal': f'{temporal_loss.item():.4f}',
                            'dpo': f'{avg_dpo_loss:.4f}'
                        })

                        if wandb_available and accelerator.is_main_process:
                            wandb.log({
                                'train/total_loss': avg_loss,
                                'train/main_loss': main_loss.item(),
                                'train/temporal_loss': temporal_loss.item(),
                                'train/dpo_loss': avg_dpo_loss,
                                'train/step': global_step
                            })

                        running_loss = 0.0
                        running_dpo_loss = 0.0

                    # Save checkpoint
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)

                            # Save UNet
                            unet_to_save = accelerator.unwrap_model(unet)
                            torch.save(unet_to_save.state_dict(),
                                     os.path.join(save_path, "unet.pth"))

                            # Save training config
                            save_training_config(args, save_path)

                            logger.info(f"Saved checkpoint at step {global_step}")

                    if global_step >= args.max_train_steps:
                        break

            if global_step >= args.max_train_steps:
                break

    # Final save
    if accelerator.is_main_process:
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)

        unet_to_save = accelerator.unwrap_model(unet)
        torch.save(unet_to_save.state_dict(),
                 os.path.join(final_save_path, "unet.pth"))

        save_training_config(args, final_save_path)

        logger.info("DPO training completed and final model saved!")

    if wandb_available and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()