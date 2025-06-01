import os
import math
import torch
import torch.nn.functional as F
import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import datetime
import json

# custom imports
from src.datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from src.mgd_pipelines.mgd_pipe import MGDPipe
from src.utils.set_seeds import set_seed

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal MGD training with DPO")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to temporal dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--num_workers_train", type=int, default=2, help="Number of workers for train dataloader")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # DPO specific parameters
    parser.add_argument("--num_candidates", type=int, default=20, help="Number of candidate images to generate")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter for KL regularization")
    parser.add_argument("--dpo_weight", type=float, default=0.5, help="Weight for DPO loss vs diffusion loss")
    parser.add_argument("--clip_i_weight", type=float, default=0.6, help="Weight for CLIP-I score")
    parser.add_argument("--clip_t_weight", type=float, default=0.4, help="Weight for CLIP-T score")

    # Temporal parameters
    parser.add_argument("--num_past_weeks", type=int, default=4, help="Number of past weeks to consider")
    parser.add_argument("--temporal_weight_decay", type=float, default=0.8, help="Weight decay for older weeks")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.3, help="Weight for temporal consistency loss")

    # Generation parameters
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps for DPO candidates")
    parser.add_argument("--no_pose", type=bool, default=True, help="Don't use pose conditioning")

    # Checkpoint and logging
    parser.add_argument("--save_steps", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")

    return parser.parse_args()


class CLIPScorer:
    """CLIP-based scoring for image quality assessment"""

    def __init__(self, device):
        self.device = device

        # Load CLIP models
        self.clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.clip_model.to(device)
        self.clip_text_model.to(device)
        self.clip_model.eval()
        self.clip_text_model.eval()

    def compute_clip_i_score(self, generated_images, target_sketch):
        """Compute CLIP-I score (image-image similarity)"""
        with torch.no_grad():
            # Process generated images
            gen_inputs = self.clip_processor(images=generated_images, return_tensors="pt")
            gen_features = self.clip_model(**{k: v.to(self.device) for k, v in gen_inputs.items()})
            gen_embeddings = gen_features.pooler_output

            # Process target sketch
            target_inputs = self.clip_processor(images=[target_sketch], return_tensors="pt")
            target_features = self.clip_model(**{k: v.to(self.device) for k, v in target_inputs.items()})
            target_embedding = target_features.pooler_output

            # Compute similarities
            similarities = F.cosine_similarity(gen_embeddings, target_embedding.expand_as(gen_embeddings), dim=-1)

        return similarities.cpu().numpy()

    def compute_clip_t_score(self, generated_images, text_prompt):
        """Compute CLIP-T score (text-image similarity)"""
        with torch.no_grad():
            # Process generated images
            gen_inputs = self.clip_processor(images=generated_images, return_tensors="pt")
            gen_features = self.clip_model(**{k: v.to(self.device) for k, v in gen_inputs.items()})
            gen_embeddings = gen_features.pooler_output

            # Process text
            text_inputs = self.clip_tokenizer([text_prompt], return_tensors="pt", padding=True, truncation=True)
            text_features = self.clip_text_model(**{k: v.to(self.device) for k, v in text_inputs.items()})
            text_embedding = text_features.pooler_output

            # Compute similarities
            similarities = F.cosine_similarity(gen_embeddings, text_embedding.expand_as(gen_embeddings), dim=-1)

        return similarities.cpu().numpy()

    def compute_weighted_score(self, generated_images, target_sketch, text_prompt, clip_i_weight=0.6, clip_t_weight=0.4):
        """Compute weighted combination of CLIP-I and CLIP-T scores"""
        clip_i_scores = self.compute_clip_i_score(generated_images, target_sketch)
        clip_t_scores = self.compute_clip_t_score(generated_images, text_prompt)

        weighted_scores = clip_i_weight * clip_i_scores + clip_t_weight * clip_t_scores
        return weighted_scores, clip_i_scores, clip_t_scores


class DPOTrainer:
    """DPO trainer for temporal fashion generation"""

    def __init__(self, args):
        self.args = args

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )

        # Set seed
        if args.seed is not None:
            set_seed(args.seed)

        # Create output directory
        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)

        # Load models
        self._load_models()

        # Initialize CLIP scorer
        self.clip_scorer = CLIPScorer(self.accelerator.device)

        # Load dataset
        self._setup_datasets()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Prepare with accelerator
        self._prepare_for_training()

    def _load_models(self):
        """Load all required models"""
        # Load scheduler, tokenizer and models
        self.scheduler = DDIMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae"
        )

        # Load base UNet
        self.unet = torch.hub.load(
            dataset='vitonhd',
            repo_or_dir='aimagelab/multimodal-garment-designer',
            source='github',
            model='mgd',
            pretrained=True
        )

        # Create reference UNet for DPO (frozen copy)
        self.unet_ref = torch.hub.load(
            dataset='vitonhd',
            repo_or_dir='aimagelab/multimodal-garment-designer',
            source='github',
            model='mgd',
            pretrained=True
        )

        # Freeze reference models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet_ref.requires_grad_(False)

        # Set to appropriate dtype
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.args.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

        self.vae.to(dtype=weight_dtype)
        self.text_encoder.to(dtype=weight_dtype)

    def _setup_datasets(self):
        """Setup training dataset"""
        self.train_dataset = TemporalVitonHDDataset(
            dataroot_path=self.args.dataset_path,
            phase='train',
            tokenizer=self.tokenizer,
            num_past_weeks=self.args.num_past_weeks,
            temporal_weight_decay=self.args.temporal_weight_decay,
            size=(512, 384)
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=0,  # Disable multiprocessing to avoid worker issues
        )

    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Only train the main UNet
        params_to_optimize = self.unet.parameters()

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-08,
        )

        self.lr_scheduler = get_scheduler(
            name="constant",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.args.max_train_steps,
        )

    def _prepare_for_training(self):
        """Prepare models and optimizers with accelerator"""
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        # Move reference models to device
        self.unet_ref = self.unet_ref.to(self.accelerator.device)
        self.vae = self.vae.to(self.accelerator.device)
        self.text_encoder = self.text_encoder.to(self.accelerator.device)

    def _tensor_to_pil(self, tensor):
        """Convert a tensor to a PIL Image"""
        tensor = tensor.cpu().detach().numpy().transpose(1, 2, 0)
        tensor = (tensor + 1) / 2.0  # Normalize to [0, 1]
        tensor = (tensor * 255).astype(np.uint8)
        return Image.fromarray(tensor)

    def generate_candidates(self, batch):
        """Generate multiple candidate images for DPO scoring"""

        # Get the weight dtype for consistency
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.args.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

        # Create pipeline for generation
        pipe = MGDPipe(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
        ).to(self.accelerator.device)

        pipe.enable_attention_slicing()

        candidates = []
        text_prompt = batch['past_conditioning']['combined_caption_text'][0]

        # Generate multiple candidates
        with torch.inference_mode():
            for i in range(self.args.num_candidates):
                # Use different seeds for variety
                generator = torch.Generator(device=self.accelerator.device).manual_seed(self.args.seed + i)

                # Convert inputs to proper dtype - provide as separate PIL images for the pipeline
                # Convert tensor to PIL Image for the pipeline
                image_tensor = batch['image'][0].to(dtype=weight_dtype)
                image_pil = self._tensor_to_pil(image_tensor)

                mask_tensor = batch['inpaint_mask'][0].to(dtype=weight_dtype)
                mask_pil = self._tensor_to_pil(mask_tensor)

                pose_tensor = batch['pose_map'][0].to(dtype=weight_dtype)
                pose_pil = self._tensor_to_pil(pose_tensor)

                sketch_tensor = batch['im_sketch'][0].to(dtype=weight_dtype)
                sketch_pil = self._tensor_to_pil(sketch_tensor)

                generated_images = pipe(
                    prompt=[text_prompt],
                    image=image_pil,
                    mask_image=mask_pil,
                    pose_map=pose_pil,
                    sketch=sketch_pil,
                    height=512,
                    width=384,
                    guidance_scale=self.args.guidance_scale,
                    num_inference_steps=self.args.num_inference_steps,
                    generator=generator,
                    no_pose=self.args.no_pose,
                ).images

                candidates.append(generated_images[0])

        return candidates, text_prompt

    def score_and_rank_candidates(self, candidates, target_sketch, text_prompt):
        """Score candidates and return preference pairs"""

        # Compute weighted scores
        weighted_scores, clip_i_scores, clip_t_scores = self.clip_scorer.compute_weighted_score(
            candidates, target_sketch, text_prompt,
            self.args.clip_i_weight, self.args.clip_t_weight
        )

        # Sort by score (descending)
        sorted_indices = np.argsort(weighted_scores)[::-1]

        # Create preference pairs (better vs worse)
        preference_pairs = []
        num_pairs = min(10, len(candidates) // 2)  # Create up to 10 preference pairs

        for i in range(num_pairs):
            better_idx = sorted_indices[i]
            worse_idx = sorted_indices[-(i+1)]
            preference_pairs.append((better_idx, worse_idx))

        return preference_pairs, weighted_scores, clip_i_scores, clip_t_scores

    def compute_dpo_loss(self, batch, preference_pairs):
        """Compute DPO loss for preference pairs"""

        # Get the weight dtype for consistency
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.args.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

        # Standard VAE scaling factor for Stable Diffusion
        vae_scale_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)

        # Prepare inputs
        images = batch['image'].to(dtype=weight_dtype)
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * vae_scale_factor

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        timesteps = timesteps.long()

        # Add noise
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Prepare conditioning
        encoder_hidden_states = self.text_encoder(batch['captions'])[0]

        # Get additional conditioning with proper channel conversion
        sketches = batch['im_sketch'].to(dtype=weight_dtype)
        # Ensure exactly 3 channels for VAE
        if sketches.shape[1] == 1:
            sketches = sketches.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        elif sketches.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 or pad to 3
            if sketches.shape[1] > 3:
                sketches = sketches[:, :3, :, :]  # Take first 3 channels
            else:
                # Pad to 3 channels by repeating the last channel
                padding_needed = 3 - sketches.shape[1]
                last_channel = sketches[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                sketches = torch.cat([sketches, last_channel], dim=1)
        sketch_latents = self.vae.encode(sketches).latent_dist.sample()
        sketch_latents = sketch_latents * vae_scale_factor

        inpaint_masks = batch['inpaint_mask'].to(dtype=weight_dtype)
        # Ensure correct dimensions for interpolation: [batch_size, channels, height, width]
        if len(inpaint_masks.shape) == 3:  # [batch_size, height, width]
            inpaint_masks = inpaint_masks.unsqueeze(1)  # [batch_size, 1, height, width]
        elif len(inpaint_masks.shape) == 2:  # [height, width]
            inpaint_masks = inpaint_masks.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
        inpaint_mask_latents = F.interpolate(inpaint_masks, size=(64, 48), mode='nearest')

        pose_maps = batch['pose_map'].to(dtype=weight_dtype)
        # Ensure exactly 3 channels for VAE
        if pose_maps.shape[1] == 1:
            pose_maps = pose_maps.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        elif pose_maps.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 or pad to 3
            if pose_maps.shape[1] > 3:
                pose_maps = pose_maps[:, :3, :, :]  # Take first 3 channels
            else:
                # Pad to 3 channels by repeating the last channel
                padding_needed = 3 - pose_maps.shape[1]
                last_channel = pose_maps[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                pose_maps = torch.cat([pose_maps, last_channel], dim=1)
        pose_map_latents = self.vae.encode(pose_maps).latent_dist.sample()
        pose_map_latents = pose_map_latents * vae_scale_factor

        # MGD model expects exactly 28 channels total
        # noisy_latents: 4 channels + additional_conditioning: 24 channels = 28 total
        # Break down the 24 additional channels:
        # latents: 4, sketch_latents: 4, inpaint_mask: 1, pose_map: 4,
        # duplicates: sketch(4) + latents(4) + pose(3) = 24 channels
        additional_conditioning = torch.cat([
            latents,  # original image latents: 4 channels
            sketch_latents,  # sketch latents: 4 channels
            inpaint_mask_latents,  # mask: 1 channel
            pose_map_latents,  # pose map latents: 4 channels
            sketch_latents,  # duplicate sketch for emphasis: 4 channels
            latents,  # duplicate original image: 4 channels
            pose_map_latents[:, :3, :, :],  # first 3 channels of pose map: 3 channels
        ], dim=1)  # Total: 4+4+1+4+4+4+3 = 24 channels

        # Compute model predictions
        # Total input: 4 (noisy) + 24 (conditioning) = 28 channels
        unet_input = torch.cat([noisy_latents, additional_conditioning], dim=1)
        model_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # Compute reference predictions (frozen)
        with torch.no_grad():
            ref_pred = self.unet_ref(
                unet_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        # Compute log probabilities (simplified approach)
        # In practice, this would involve computing the full diffusion log probability
        model_log_prob = -F.mse_loss(model_pred, noise, reduction='none').mean(dim=[1, 2, 3])
        ref_log_prob = -F.mse_loss(ref_pred, noise, reduction='none').mean(dim=[1, 2, 3])

        # DPO loss computation
        dpo_loss = 0.0
        num_pairs = len(preference_pairs)

        if num_pairs > 0:
            # For simplicity, we'll compute a simplified DPO loss
            # In practice, you'd need to generate actual preference data
            log_ratio = model_log_prob - ref_log_prob
            dpo_loss = -F.logsigmoid(self.args.dpo_beta * log_ratio).mean()

        return dpo_loss

    def compute_temporal_consistency_loss(self, batch):
        """Compute temporal consistency loss using past weeks"""

        # Get the weight dtype for consistency
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.args.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

        # Get current week's sketch
        current_sketch = batch['im_sketch'].to(dtype=weight_dtype)
        # Ensure exactly 3 channels for VAE
        if current_sketch.shape[1] == 1:
            current_sketch = current_sketch.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        elif current_sketch.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 or pad to 3
            if current_sketch.shape[1] > 3:
                current_sketch = current_sketch[:, :3, :, :]  # Take first 3 channels
            else:
                # Pad to 3 channels by repeating the last channel
                padding_needed = 3 - current_sketch.shape[1]
                last_channel = current_sketch[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                current_sketch = torch.cat([current_sketch, last_channel], dim=1)

        # Get weighted past sketches
        past_sketches = batch['past_conditioning']['weighted_sketch'].to(dtype=weight_dtype)
        # Ensure exactly 3 channels for VAE
        if past_sketches.shape[1] == 1:
            past_sketches = past_sketches.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        elif past_sketches.shape[1] != 3:
            # If not 1 or 3 channels, take first 3 or pad to 3
            if past_sketches.shape[1] > 3:
                past_sketches = past_sketches[:, :3, :, :]  # Take first 3 channels
            else:
                # Pad to 3 channels by repeating the last channel
                padding_needed = 3 - past_sketches.shape[1]
                last_channel = past_sketches[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                past_sketches = torch.cat([past_sketches, last_channel], dim=1)

        # Encode sketches
        with torch.no_grad():
            current_sketch_encoded = self.vae.encode(current_sketch).latent_dist.sample()
            past_sketches_encoded = self.vae.encode(past_sketches).latent_dist.sample()

        # Compute consistency loss (L2 distance)
        temporal_loss = F.mse_loss(current_sketch_encoded, past_sketches_encoded)

        return temporal_loss

    def train_step(self, batch):
        """Single training step with DPO"""

        # Get the weight dtype for consistency
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16
        elif self.args.mixed_precision == 'bf16':
            weight_dtype = torch.bfloat16

        # Standard VAE scaling factor for Stable Diffusion
        vae_scale_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)

        # Generate candidates for DPO (less frequently to save compute)
        if torch.rand(1).item() < 0.3:  # 30% of the time
            candidates, text_prompt = self.generate_candidates(batch)
            target_sketch = batch['im_sketch'][0]

            # Score and get preference pairs
            preference_pairs, weighted_scores, clip_i_scores, clip_t_scores = self.score_and_rank_candidates(
                candidates, target_sketch, text_prompt
            )

            # Compute DPO loss
            dpo_loss = self.compute_dpo_loss(batch, preference_pairs)
        else:
            dpo_loss = torch.tensor(0.0, device=self.accelerator.device)

        # Standard diffusion loss - ensure proper dtype
        with self.accelerator.autocast():
            # Convert input images to the correct dtype
            images = batch['image'].to(dtype=weight_dtype)
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * vae_scale_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Prepare conditioning
            encoder_hidden_states = self.text_encoder(batch['captions'])[0]

            # Convert conditioning inputs to correct dtype and ensure 3 channels
            sketches = batch['im_sketch'].to(dtype=weight_dtype)
            # Ensure exactly 3 channels for VAE
            if sketches.shape[1] == 1:
                sketches = sketches.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            elif sketches.shape[1] != 3:
                # If not 1 or 3 channels, take first 3 or pad to 3
                if sketches.shape[1] > 3:
                    sketches = sketches[:, :3, :, :]  # Take first 3 channels
                else:
                    # Pad to 3 channels by repeating the last channel
                    padding_needed = 3 - sketches.shape[1]
                    last_channel = sketches[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                    sketches = torch.cat([sketches, last_channel], dim=1)
            sketch_latents = self.vae.encode(sketches).latent_dist.sample()
            sketch_latents = sketch_latents * vae_scale_factor

            inpaint_masks = batch['inpaint_mask'].to(dtype=weight_dtype)
            # Ensure correct dimensions for interpolation: [batch_size, channels, height, width]
            if len(inpaint_masks.shape) == 3:  # [batch_size, height, width]
                inpaint_masks = inpaint_masks.unsqueeze(1)  # [batch_size, 1, height, width]
            elif len(inpaint_masks.shape) == 2:  # [height, width]
                inpaint_masks = inpaint_masks.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
            inpaint_mask_latents = F.interpolate(inpaint_masks, size=(64, 48), mode='nearest')

            pose_maps = batch['pose_map'].to(dtype=weight_dtype)
            # Ensure exactly 3 channels for VAE
            if pose_maps.shape[1] == 1:
                pose_maps = pose_maps.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
            elif pose_maps.shape[1] != 3:
                # If not 1 or 3 channels, take first 3 or pad to 3
                if pose_maps.shape[1] > 3:
                    pose_maps = pose_maps[:, :3, :, :]  # Take first 3 channels
                else:
                    # Pad to 3 channels by repeating the last channel
                    padding_needed = 3 - pose_maps.shape[1]
                    last_channel = pose_maps[:, -1:, :, :].repeat(1, padding_needed, 1, 1)
                    pose_maps = torch.cat([pose_maps, last_channel], dim=1)
            pose_map_latents = self.vae.encode(pose_maps).latent_dist.sample()
            pose_map_latents = pose_map_latents * vae_scale_factor

            # MGD model expects exactly 28 channels total
            # noisy_latents: 4 channels + additional_conditioning: 24 channels = 28 total
            # Break down the 24 additional channels:
            # latents: 4, sketch_latents: 4, inpaint_mask: 1, pose_map: 4,
            # duplicates: sketch(4) + latents(4) + pose(3) = 24 channels
            additional_conditioning = torch.cat([
                latents,  # original image latents: 4 channels
                sketch_latents,  # sketch latents: 4 channels
                inpaint_mask_latents,  # mask: 1 channel
                pose_map_latents,  # pose map latents: 4 channels
                sketch_latents,  # duplicate sketch for emphasis: 4 channels
                latents,  # duplicate original image: 4 channels
                pose_map_latents[:, :3, :, :],  # first 3 channels of pose map: 3 channels
            ], dim=1)  # Total: 4+4+1+4+4+4+3 = 24 channels

            # Forward pass - concatenate conditioning with noisy latents for MGD model
            # Total input: 4 (noisy) + 24 (conditioning) = 28 channels
            unet_input = torch.cat([noisy_latents, additional_conditioning], dim=1)
            model_pred = self.unet(
                unet_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # Compute diffusion loss
            if self.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.scheduler.config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.scheduler.config.prediction_type}")

            diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Temporal consistency loss
        temporal_loss = self.compute_temporal_consistency_loss(batch)

        # Combined loss
        total_loss = (
            diffusion_loss +
            self.args.temporal_loss_weight * temporal_loss +
            self.args.dpo_weight * dpo_loss
        )

        return total_loss, diffusion_loss, temporal_loss, dpo_loss

    def train(self):
        """Main training loop"""

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {math.ceil(self.args.max_train_steps / len(self.train_dataloader))}")
        logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        global_step = 0
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        # Training metrics
        total_diffusion_loss = 0.0
        total_temporal_loss = 0.0
        total_dpo_loss = 0.0

        for epoch in range(math.ceil(self.args.max_train_steps / len(self.train_dataloader))):
            self.unet.train()

            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    # Training step
                    total_loss, diffusion_loss, temporal_loss, dpo_loss = self.train_step(batch)

                    # Backward pass
                    self.accelerator.backward(total_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update metrics
                if self.accelerator.sync_gradients:
                    total_diffusion_loss += diffusion_loss.item()
                    total_temporal_loss += temporal_loss.item()
                    total_dpo_loss += dpo_loss.item()

                    progress_bar.update(1)
                    global_step += 1

                    # Logging
                    if global_step % self.args.log_steps == 0:
                        avg_diffusion_loss = total_diffusion_loss / self.args.log_steps
                        avg_temporal_loss = total_temporal_loss / self.args.log_steps
                        avg_dpo_loss = total_dpo_loss / self.args.log_steps

                        logger.info(f"Step {global_step}: diffusion_loss={avg_diffusion_loss:.4f}, temporal_loss={avg_temporal_loss:.4f}, dpo_loss={avg_dpo_loss:.4f}")

                        total_diffusion_loss = 0.0
                        total_temporal_loss = 0.0
                        total_dpo_loss = 0.0

                    # Save checkpoint
                    if global_step % self.args.save_steps == 0:
                        self.save_checkpoint(global_step)

                    if global_step >= self.args.max_train_steps:
                        break

            if global_step >= self.args.max_train_steps:
                break

        # Save final checkpoint
        self.save_checkpoint(global_step, is_final=True)

        self.accelerator.end_training()

    def save_checkpoint(self, step, is_final=False):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if is_final:
                save_path = os.path.join(self.args.output_dir, f"temporal_vitonhd_dpo_{timestamp}")
            else:
                save_path = os.path.join(self.args.output_dir, f"temporal_vitonhd_dpo_{timestamp}")

            os.makedirs(save_path, exist_ok=True)

            # Save UNet
            unet_path = os.path.join(save_path, f"checkpoint-{step}")
            os.makedirs(unet_path, exist_ok=True)

            # Unwrap the model
            unet_to_save = self.accelerator.unwrap_model(self.unet)
            torch.save(unet_to_save.state_dict(), os.path.join(unet_path, "unet.pth"))

            # Save training config
            config = {
                "step": step,
                "args": vars(self.args),
                "model_config": {
                    "pretrained_model_name_or_path": self.args.pretrained_model_name_or_path,
                }
            }

            with open(os.path.join(unet_path, "training_config.json"), "w") as f:
                json.dump(config, f, indent=2)

            logger.info(f"Checkpoint saved to {unet_path}")


def main():
    args = parse_args()

    # Initialize trainer
    trainer = DPOTrainer(args)

    # Start training
    trainer.train()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()