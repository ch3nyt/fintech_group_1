import os
import torch
import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F

# custom imports
from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from utils.set_seeds import set_seed
from utils.image_from_pipe import generate_images_from_mgd_pipe
from utils.garment_classifier import (
    classify_garment,
    ensure_garment_consistency
)

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal MGD evaluation")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="alwold/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to temporal dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained temporal UNet")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")

    # Temporal parameters
    parser.add_argument("--num_past_weeks", type=int, default=4, help="Number of past weeks to consider")
    parser.add_argument("--temporal_weight_decay", type=float, default=0.8, help="Weight decay for older weeks")
    parser.add_argument("--predict_future_weeks", type=int, default=1, help="Number of future weeks to predict")

    # Category filter
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                       help="Categories to evaluate on. If not specified, use all categories")

    # Inference parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers_test", type=int, default=2, help="Number of workers for test dataloader")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_pose", type=bool, default=True, help="Don't use pose conditioning")

    return parser.parse_args()


class GarmentSegmentationModel:
    """Simple garment segmentation model using pre-trained models"""

    def __init__(self, device="cuda"):
        self.device = device
        # Use a simple approach - could be replaced with more sophisticated models
        self.setup_model()

    def setup_model(self):
        """Setup segmentation model - using simple thresholding for now"""
        # This is a placeholder - in practice you'd load a proper segmentation model
        # For now, we'll use image processing techniques
        logger.info("Setting up garment segmentation model...")

    def segment_garment(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Segment garment from image
        Args:
            image_tensor: [C, H, W] image tensor in range [-1, 1]
        Returns:
            mask: [H, W] binary mask where 1 = garment region
        """
        # Convert to PIL for processing
        image_np = ((image_tensor + 1) / 2).clamp(0, 1)
        image_pil = transforms.ToPILImage()(image_np.cpu())

        # Convert to numpy for processing
        img_np = np.array(image_pil)

        # Create garment mask using color and region analysis
        mask = self._create_garment_mask(img_np)

        return torch.from_numpy(mask).float()

    def _create_garment_mask(self, img_np: np.ndarray) -> np.ndarray:
        """Create garment mask using image processing techniques"""
        h, w = img_np.shape[:2]

        # Create a center-focused mask that covers typical garment regions
        mask = np.zeros((h, w), dtype=np.float32)

        # Define garment region based on typical clothing positions
        # Top region (for tops, dresses)
        top_region = slice(int(h * 0.2), int(h * 0.7))
        top_width = slice(int(w * 0.15), int(w * 0.85))

        # Bottom region (for pants, skirts)
        bottom_region = slice(int(h * 0.45), int(h * 0.9))
        bottom_width = slice(int(w * 0.25), int(w * 0.75))

        # Analyze image brightness and color to refine mask
        gray = np.mean(img_np, axis=2)

        # Use edge detection to find garment boundaries
        from scipy import ndimage

        # Simple edge detection
        edges = ndimage.sobel(gray)
        edges = (edges > np.percentile(edges, 70)).astype(float)

        # Combine geometric regions with edge information
        mask[top_region, top_width] = 1.0
        mask[bottom_region, bottom_width] = 1.0

        # Refine with edge information
        mask = mask * (1 - edges * 0.3)  # Reduce mask at strong edges

        # Smooth the mask
        mask = ndimage.gaussian_filter(mask, sigma=2)

        # Threshold to create binary mask
        mask = (mask > 0.3).astype(np.float32)

        # Ensure connected regions
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5))).astype(np.float32)

        return mask

    def create_focused_inpaint_mask(self, image_tensor: torch.Tensor, garment_category: str = None) -> torch.Tensor:
        """
        Create a focused inpaint mask based on garment type
        Args:
            image_tensor: [C, H, W] image tensor
            garment_category: Type of garment for focused masking
        Returns:
            mask: [1, H, W] inpaint mask
        """
        # Get base garment segmentation
        garment_mask = self.segment_garment(image_tensor)

        h, w = garment_mask.shape

        # Create category-specific masks
        if garment_category == 'top':
            # Focus on upper body region
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.15):int(h*0.65), int(w*0.1):int(w*0.9)] = 1.0

        elif garment_category == 'dress':
            # Focus on torso and upper legs
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.15):int(h*0.8), int(w*0.15):int(w*0.85)] = 1.0

        elif garment_category == 'pants':
            # Focus on lower body region
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.4):int(h*0.9), int(w*0.2):int(w*0.8)] = 1.0

        elif garment_category == 'shoes':
            # Focus on feet region
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.8):int(h*1.0), int(w*0.2):int(w*0.8)] = 1.0

        elif garment_category == 'underwear':
            # Focus on torso region
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.2):int(h*0.6), int(w*0.25):int(w*0.75)] = 1.0

        else:
            # Default: center region
            region_mask = torch.zeros_like(garment_mask)
            region_mask[int(h*0.2):int(h*0.8), int(w*0.15):int(w*0.85)] = 1.0

        # Combine garment mask with region mask
        focused_mask = garment_mask * region_mask

        # Dilate slightly to ensure coverage
        kernel_size = 7
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)

        # Dilate the mask
        expanded_mask = F.conv2d(
            focused_mask.unsqueeze(0).unsqueeze(0).float(),
            kernel,
            padding=kernel_size//2
        )

        # Threshold and return
        final_mask = (expanded_mask > 0.1).float()

        return final_mask.squeeze(0)  # Return [1, H, W]


class TemporalPredictor:
    """
    Class for predicting future week garments based on temporal patterns
    """

    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(mixed_precision=args.mixed_precision)
        self.device = self.accelerator.device

        # Set seed
        if args.seed is not None:
            set_seed(args.seed)

        # Initialize segmentation model
        self.segmentation_model = GarmentSegmentationModel(device=self.device)

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all required models"""
        # Load scheduler, tokenizer and models for SD2 inpainting
        self.scheduler = DDIMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(self.args.num_inference_steps, device=self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae"
        )

        # Load SD2 inpainting UNet directly instead of custom MGD UNet
        from diffusers import UNet2DConditionModel
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="unet"
        )

        logger.info("Using Stable Diffusion 2 inpainting UNet (9 channels)")
        logger.info(f"UNet input channels: {self.unet.config.in_channels}")

        # Load checkpoint if available and compatible
        if os.path.exists(self.args.checkpoint_path):
            logger.info(f"Loading checkpoint from {self.args.checkpoint_path}")
            try:
                # Load the checkpoint state dict
                checkpoint_state_dict = torch.load(self.args.checkpoint_path, map_location="cpu")

                # Load into the UNet
                self.unet.load_state_dict(checkpoint_state_dict)
                logger.info("✅ Successfully loaded SD2-compatible checkpoint!")
                logger.info(f"Checkpoint loaded from: {self.args.checkpoint_path}")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.warning("Falling back to pretrained SD2 inpainting weights")
        else:
            logger.info(f"No checkpoint found at {self.args.checkpoint_path}")
            logger.info("Using pretrained SD2 inpainting weights only")

        # Freeze models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.eval()

        # Move to device
        weight_dtype = torch.float32
        if self.args.mixed_precision == 'fp16':
            weight_dtype = torch.float16

        self.text_encoder.to(self.device, dtype=weight_dtype)
        self.vae.to(self.device, dtype=weight_dtype)
        self.unet.to(self.device, dtype=weight_dtype)

    def predict_next_week(self, past_weeks_data, next_week_actual_text=None):
        """Predict next week's garment using temporal context"""

        # Create SD2 inpainting pipeline instead of MGD pipeline
        from diffusers import StableDiffusionInpaintPipeline

        pipe = StableDiffusionInpaintPipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
            safety_checker=None,  # Disable safety checker for speed
            feature_extractor=None,
        ).to(self.device)

        pipe.enable_attention_slicing()

        predictions = []

        with torch.inference_mode():
            for week_data in past_weeks_data:
                past_conditioning = week_data['past_conditioning']
                current_week_text = past_conditioning['combined_caption_text']

                # Use garment classifier to determine category and ensure consistency
                current_garment_category = classify_garment(current_week_text)
                print(f"🔍 Current week garment category: {current_garment_category}")

                # Determine style prompt (what text to use for generation)
                base_text_for_evolution = past_conditioning['combined_caption_text']
                print(f"📜 Base text for evolution: '{base_text_for_evolution}'")

                if next_week_actual_text is not None and next_week_actual_text.strip():
                    # Use the actual next week text directly without any modifications
                    next_week_style = next_week_actual_text.strip()
                    print(f"✅ Using actual next week text directly: '{next_week_style}'")
                    print(f"🚫 NO modifications applied - using raw actual text")
                    evolution_source = "actual_next_week_text_direct"
                    evolution_base_text = next_week_actual_text.strip()
                    # Record which product is providing the actual next week text
                    actual_text_source_product = week_data.get('next_week_source_product', 'unknown')
                    actual_text_source_week = week_data.get('next_week_source_week', 'unknown')
                    print(f"📋 Actual text source product: {actual_text_source_product}")
                    print(f"📅 Actual text source week info: {actual_text_source_week}")
                else:
                    # Generate evolved style based on past trends
                    base_style = past_conditioning['combined_caption_text']
                    evolution_source = "evolved_from_past_trends"
                    evolution_base_text = base_style

                    # Enhanced text evolution logic with category consistency
                    style_keywords = ["trending", "modern", "stylish", "contemporary", "fashionable"]
                    color_variations = ["refined", "elegant", "sophisticated", "updated", "enhanced"]

                    # Extract key elements from base style
                    base_lower = base_style.lower()
                    evolved_elements = []

                    # Add trending modifier
                    evolved_elements.append(f"trending {color_variations[hash(base_style) % len(color_variations)]}")

                    # Preserve core garment type but evolve style - use classifier to ensure consistency
                    if current_garment_category:
                        if current_garment_category == 'top':
                            evolved_elements.append("fashionable top")
                        elif current_garment_category == 'dress':
                            evolved_elements.append("stylish dress")
                        elif current_garment_category == 'pants':
                            evolved_elements.append("modern pants")
                        elif current_garment_category == 'shoes':
                            evolved_elements.append("contemporary shoes")
                        elif current_garment_category == 'underwear':
                            evolved_elements.append("refined underwear")
                        elif current_garment_category == 'skirt':
                            evolved_elements.append("elegant skirt")
                        elif current_garment_category == 'shorts':
                            evolved_elements.append("stylish shorts")
                        elif current_garment_category == 'outerwear':
                            evolved_elements.append("modern jacket")
                        else:
                            evolved_elements.append("updated garment")
                    else:
                        # Fallback to original logic
                        if "top" in base_lower or "shirt" in base_lower or "blouse" in base_lower:
                            evolved_elements.append("fashionable top")
                        elif "dress" in base_lower:
                            evolved_elements.append("stylish dress")
                        elif "pants" in base_lower or "trousers" in base_lower:
                            evolved_elements.append("modern pants")
                        elif "shoe" in base_lower or "boot" in base_lower:
                            evolved_elements.append("contemporary footwear")
                        elif "underwear" in base_lower:
                            evolved_elements.append("refined undergarment")
                        else:
                            evolved_elements.append("updated garment")

                    # Add color/material evolution
                    if "black" in base_lower:
                        evolved_elements.append("with sophisticated black styling")
                    elif "white" in base_lower:
                        evolved_elements.append("with elegant white tones")
                    elif "blue" in base_lower:
                        evolved_elements.append("with modern blue accents")
                    else:
                        evolved_elements.append("with contemporary design elements")

                    next_week_style = " ".join(evolved_elements)
                    print(f"🔮 Generated evolved style: '{next_week_style}'")
                    print(f"🎯 Based on: '{base_style}'")

                # Log the evolution chain
                print(f"🔗 Evolution chain:")
                print(f"   📍 Source: {evolution_source}")
                print(f"   📜 Base text: '{evolution_base_text}'")
                print(f"   ➡️  Generated: '{next_week_style}'")

                if next_week_actual_text is not None and next_week_actual_text.strip():
                    print(f"📅 Using next week text: '{next_week_style}'")
                else:
                    print(f"🔮 Evolved text: '{next_week_style}'")

                # MODIFIED: Create precise segmentation-based inpaint mask
                original_mask = week_data['inpaint_mask']
                print(f"🎭 Original mask shape: {original_mask.shape}")
                print(f"🎭 Original mask range: [{original_mask.min():.3f}, {original_mask.max():.3f}]")

                # Use segmentation model to create precise garment mask
                current_image = week_data['image']  # [C, H, W]

                # Get garment category for focused masking
                garment_category = current_garment_category if current_garment_category else 'general'

                print(f"🤖 Running segmentation model for category: {garment_category}")

                # Generate precise inpaint mask using segmentation
                segmentation_mask = self.segmentation_model.create_focused_inpaint_mask(
                    current_image,
                    garment_category=garment_category
                )

                # Move to correct device
                segmentation_mask = segmentation_mask.to(current_image.device)

                print(f"🎨 Segmentation mask shape: {segmentation_mask.shape}")
                print(f"🎨 Segmentation mask range: [{segmentation_mask.min():.3f}, {segmentation_mask.max():.3f}]")
                print(f"🎨 Mask area ratio: {segmentation_mask.sum() / segmentation_mask.numel():.3f}")
                print("🎯 Using SEGMENTATION-BASED mask - precise garment regions")

                # Convert tensors to PIL Images for SD2 inpainting pipeline
                with self.accelerator.autocast():
                    print(f"🎯 Using SD2 inpainting pipeline")
                    print(f"🎨 Generating image based ONLY on evolved text (ignoring base style)")
                    if next_week_actual_text is not None and next_week_actual_text.strip():
                        print(f"📅 Final generation prompt: '{next_week_style}'")
                    else:
                        print(f"🔮 Final generation prompt: '{next_week_style}'")
                    print("🔄 Generating with SD2 inpainting (text-only guidance)...")

                    # Convert image tensor to PIL
                    image_pil = transforms.ToPILImage()(((current_image + 1) / 2).clamp(0, 1).cpu())

                    # Convert mask tensor to PIL
                    if segmentation_mask.dim() == 3 and segmentation_mask.shape[0] == 1:
                        mask_tensor = segmentation_mask.squeeze(0)
                    else:
                        mask_tensor = segmentation_mask
                    mask_pil = transforms.ToPILImage()(mask_tensor.cpu())

                    # Generate with SD2 inpainting pipeline using ONLY the evolved text
                    # No reference to base style - pure text-to-image generation in masked region
                    generated_images = pipe(
                        prompt=next_week_style,  # ONLY use evolved text, ignore base style
                        image=image_pil,
                        mask_image=mask_pil,
                        height=512,
                        width=384,
                        guidance_scale=self.args.guidance_scale,
                        num_inference_steps=self.args.num_inference_steps,
                        strength=0.99,  # High strength for significant changes
                    ).images

                    print("✅ SD2 text-only inpainting completed!")

                    predictions.append({
                        'generated_image': generated_images[0],
                        'style_prompt': next_week_style,
                        'base_style': past_conditioning['combined_caption_text'],  # Keep for logging only
                        'base_image': week_data['im_name'],
                        'temporal_weights': week_data['temporal_weights'],
                        'category': week_data['category'],
                        'product_id': week_data['product_id'],
                        'is_actual_next_week_text': next_week_actual_text is not None and next_week_actual_text.strip() != "",
                        'mask_type': 'segmentation_based',
                        'garment_category_detected': garment_category,
                        'pipeline_used': 'SD2_inpainting_text_only',
                        'generation_method': 'text_only_no_base_style',
                        'evolution_source': evolution_source,
                        'evolution_base_text': evolution_base_text,
                        'base_text_for_evolution': base_text_for_evolution,
                        'actual_text_source_product': actual_text_source_product if 'actual_text_source_product' in locals() else 'N/A',
                        'actual_text_source_week': actual_text_source_week if 'actual_text_source_week' in locals() else 'N/A'
                    })

        return predictions

    def evaluate_dataset(self):
        """Evaluate on test dataset and generate predictions"""

        # Load test dataset
        test_dataset = TemporalVitonHDDataset(
            dataroot_path=self.args.dataset_path,
            phase='test',
            tokenizer=self.tokenizer,
            num_past_weeks=self.args.num_past_weeks,
            temporal_weight_decay=self.args.temporal_weight_decay,
            size=(512, 384),
            category_filter=self.args.categories
        )

        test_dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            num_workers=0,  # Disable multiprocessing to avoid worker issues
        )

        test_dataloader = self.accelerator.prepare(test_dataloader)

        # Create output directory structure
        os.makedirs(self.args.output_dir, exist_ok=True)
        predictions_dir = os.path.join(self.args.output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Create caption log file
        caption_log_path = os.path.join(self.args.output_dir, "captions_used.txt")
        caption_log = open(caption_log_path, 'w', encoding='utf-8')
        caption_log.write("Temporal VITONhd Inference - Captions Used\n")
        caption_log.write("=" * 50 + "\n\n")

        # Create category subdirectories
        categories = self.args.categories if self.args.categories else test_dataset.CATEGORIES
        for category in categories:
            os.makedirs(os.path.join(predictions_dir, category), exist_ok=True)

        all_predictions = []

        logger.info("Starting temporal prediction evaluation...")
        logger.info(f"Evaluating on categories: {categories}")

        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
            print(f"\n{'='*60}")
            print(f"🔄 Processing batch {batch_idx + 1}/{len(test_dataloader)}")
            print(f"📁 Base image: {batch['im_name'][0]}")

            # Decode the original caption from tokens to text
            if 'original_captions' in batch:
                # Use the text caption directly from dataset
                print(f"📝 Original caption: '{batch['original_captions'][0]}'")
            elif 'captions' in batch:
                try:
                    # Decode the tokenized caption back to text
                    original_caption_text = self.tokenizer.decode(batch['captions'][0], skip_special_tokens=True)
                    print(f"📝 Original caption: '{original_caption_text}'")
                except:
                    print(f"📝 Original caption: [Unable to decode]")
            else:
                print(f"📝 Original caption: N/A")

            # Show next week information
            if 'next_week_actual_text' in batch and batch['next_week_actual_text'][0] and batch['next_week_actual_text'][0].strip():
                print(f"📅 Next week's actual text available: '{batch['next_week_actual_text'][0]}'")
                next_week_source_product = batch.get('next_week_item_id', ['unknown'])[0]
                next_week_source_week = batch.get('next_week', ['unknown'])[0]
                print(f"📋 Next week source product: {next_week_source_product}")
                print(f"📅 Next week source week: {next_week_source_week}")
                using_actual_text = True
            else:
                print(f"⚠️  No next week data available - will use evolved text as fallback")
                next_week_source_product = "N/A"
                next_week_source_week = "N/A"
                using_actual_text = False

            # Predict next week garments
            predictions = self.predict_next_week([{
                'image': batch['image'][0],
                'inpaint_mask': batch['inpaint_mask'][0],
                'pose_map': batch['pose_map'][0],
                'im_sketch': batch['im_sketch'][0],  # Current week's target sketch
                'im_seg': batch['im_seg'][0] if 'im_seg' in batch else None,  # Current week's segmentation
                'past_conditioning': {
                    'weighted_sketch': batch['past_conditioning']['weighted_sketch'][0],
                    'combined_caption_text': batch['past_conditioning']['combined_caption_text'][0]
                },
                'im_name': batch['im_name'][0],
                'temporal_weights': batch['temporal_weights'][0],
                'category': batch['category'][0],
                'product_id': batch['product_id'][0],
                'next_week_source_product': next_week_source_product,
                'next_week_source_week': next_week_source_week
            }], batch['next_week_actual_text'][0] if 'next_week_actual_text' in batch and batch['next_week_actual_text'][0].strip() else None)

            # Save predictions
            for pred_idx, prediction in enumerate(predictions):
                category = prediction['category']
                save_name = f"{prediction['product_id']}_pred_{batch_idx:04d}.jpg"
                save_path = os.path.join(predictions_dir, category, save_name)

                prediction['generated_image'].save(save_path)

                print(f"💾 Saved: {save_path}")
                print(f"📝 Final caption used: '{prediction['style_prompt']}'")

                # Write to caption log
                caption_log.write(f"Image: {save_name}\n")
                caption_log.write(f"Product ID: {prediction['product_id']}\n")
                caption_log.write(f"Category: {category}\n")
                caption_log.write(f"Approach: SD2 inpainting + Text-only generation (no base style)\n")
                caption_log.write(f"Pipeline: {prediction['pipeline_used']}\n")
                caption_log.write(f"Generation Method: {prediction['generation_method']}\n")
                caption_log.write(f"Mask Type: {prediction['mask_type']} (segmentation-based)\n")
                caption_log.write(f"Detected Garment Category: {prediction['garment_category_detected']}\n")
                caption_log.write(f"Base Style (for reference only): {prediction['base_style']}\n")
                caption_log.write(f"Final Generation Prompt: {prediction['style_prompt']}\n")
                caption_log.write(f"Temporal Weights: {prediction['temporal_weights']}\n")
                caption_log.write(f"Evolution Source: {prediction['evolution_source']}\n")
                caption_log.write(f"Evolution Base Text: {prediction['evolution_base_text']}\n")
                caption_log.write(f"Text Used for Evolution: {prediction['base_text_for_evolution']}\n")
                caption_log.write(f"Actual Text Source Product: {prediction['actual_text_source_product']}\n")
                caption_log.write(f"Actual Text Source Week: {prediction['actual_text_source_week']}\n")
                caption_log.write("-" * 40 + "\n\n")
                caption_log.flush()

                # Store metadata
                all_predictions.append({
                    'prediction_file': os.path.join(category, save_name),
                    'style_prompt': prediction['style_prompt'],
                    'base_style': prediction['base_style'],
                    'base_image': prediction['base_image'],
                    'temporal_weights': prediction['temporal_weights'].tolist(),
                    'category': category,
                    'product_id': prediction['product_id'],
                    'is_actual_next_week_text': prediction['is_actual_next_week_text'],
                    'mask_type': prediction['mask_type'],
                    'garment_category_detected': prediction['garment_category_detected'],
                    'pipeline_used': prediction['pipeline_used'],
                    'generation_method': prediction['generation_method'],
                    'evolution_source': prediction['evolution_source'],
                    'evolution_base_text': prediction['evolution_base_text'],
                    'base_text_for_evolution': prediction['base_text_for_evolution'],
                    'actual_text_source_product': prediction['actual_text_source_product'],
                    'actual_text_source_week': prediction['actual_text_source_week']
                })

        # Save prediction metadata
        with open(os.path.join(self.args.output_dir, 'predictions_metadata.json'), 'w') as f:
            json.dump(all_predictions, f, indent=2)

        # Save category statistics
        category_stats = {}
        for pred in all_predictions:
            cat = pred['category']
            if cat not in category_stats:
                category_stats[cat] = 0
            category_stats[cat] += 1

        with open(os.path.join(self.args.output_dir, 'category_statistics.json'), 'w') as f:
            json.dump(category_stats, f, indent=2)

        # Close caption log
        caption_log.write(f"\nTotal predictions generated: {len(all_predictions)}\n")
        caption_log.write(f"Category distribution: {category_stats}\n")
        caption_log.close()

        logger.info(f"Generated {len(all_predictions)} predictions")
        logger.info(f"Category distribution: {category_stats}")
        logger.info(f"Results saved to {self.args.output_dir}")
        logger.info(f"Caption log saved to {caption_log_path}")

        return all_predictions


def main():
    args = parse_args()

    # Initialize predictor
    predictor = TemporalPredictor(args)

    # Run evaluation
    predictions = predictor.evaluate_dataset()

    logger.info("Temporal prediction evaluation completed!")


if __name__ == "__main__":
    main()