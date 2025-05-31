import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import numpy as np
from typing import Optional, Union, List
import wandb

# custom imports
from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from utils.set_seeds import set_seed

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="Temporal MGD training")

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

    # Category filter
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="Categories to train on (e.g., top5acc top5gfb). If not specified, use all categories")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Loss parameters
    parser.add_argument("--noise_offset", type=float, default=0.1, help="Noise offset for training")
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="SNR weighting gamma")

    # Logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--project_name", type=str, default="temporal-mgd", help="Wandb project name")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")

    return parser.parse_args()


def compute_temporal_consistency_loss(current_latents, past_conditioning, temporal_weights):
    """
    Compute temporal consistency loss to encourage smooth transitions
    """
    # Extract weighted features from past conditioning
    weighted_past_features = past_conditioning['weighted_sketch']

    # Downsample current latents to match conditioning size
    current_features = F.interpolate(
        current_latents,
        size=weighted_past_features.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    # Compute L2 distance weighted by temporal weights
    consistency_loss = F.mse_loss(current_features, weighted_past_features.unsqueeze(0))

    # Weight by temporal importance (more recent should have higher influence)
    temporal_weight = temporal_weights.max()  # Use highest weight (most recent)

    return consistency_loss * temporal_weight


def compute_snr_weights(timesteps, noise_scheduler, gamma=5.0):
    """
    Compute signal-to-noise ratio weights for loss reweighting
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

    # SNR = signal^2 / noise^2
    snr = (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2
    snr_weights = torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]

    return snr_weights


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
        wandb.init(project=args.project_name, config=vars(args))

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

    # Load UNet - start from pretrained MGD model
    unet = torch.hub.load(
        dataset='vitonhd',  # or your preferred base dataset
        repo_or_dir='aimagelab/multimodal-garment-designer',
        source='github',
        model='mgd',
        pretrained=True
    )

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Only train UNet
    unet.requires_grad_(True)

    # Enable memory efficient attention
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # Enable gradient checkpointing to save memory
    unet.enable_gradient_checkpointing()

    # Prepare dataset
    train_dataset = TemporalVitonHDDataset(
        dataroot_path=args.dataset_path,
        phase='train',
        tokenizer=tokenizer,
        num_past_weeks=args.num_past_weeks,
        temporal_weight_decay=args.temporal_weight_decay,
        size=(512, 384),
        category_filter=args.categories  # Filter specific categories if provided
    )

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
    # Note: scheduler doesn't need .to() method - it's not a PyTorch module

    # Enable sequential CPU offload for memory efficiency
    # Note: This is experimental and might slow down training
    # Uncomment if you need extreme memory savings
    # if hasattr(unet, 'enable_sequential_cpu_offload'):
    #     unet.enable_sequential_cpu_offload()

    # Training loop
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Training Steps")

    global_step = 0
    running_loss = 0.0

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
                # Use standard VAE scaling factor (0.18215 for Stable Diffusion VAE)
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
                # Resize conditioning to latent space
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

                # Total loss
                total_loss = main_loss + args.temporal_loss_weight * temporal_loss

                # Backward pass
                accelerator.backward(total_loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Logging
                running_loss += total_loss.item()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % 50 == 0:
                        avg_loss = running_loss / 50
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'main': f'{main_loss.item():.4f}',
                            'temporal': f'{temporal_loss.item():.4f}'
                        })

                        if args.use_wandb and accelerator.is_main_process:
                            wandb.log({
                                'train/total_loss': avg_loss,
                                'train/main_loss': main_loss.item(),
                                'train/temporal_loss': temporal_loss.item(),
                                'train/step': global_step
                            })

                        running_loss = 0.0

                    # Save checkpoint
                    if global_step % args.save_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)

                            # Save UNet
                            unet_to_save = accelerator.unwrap_model(unet)
                            torch.save(unet_to_save.state_dict(),
                                     os.path.join(save_path, "unet.pth"))

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

        logger.info("Training completed and final model saved!")

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()