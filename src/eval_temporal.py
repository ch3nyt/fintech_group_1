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

# custom imports
from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from utils.set_seeds import set_seed
from utils.image_from_pipe import generate_images_from_mgd_pipe

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal MGD evaluation")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
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

        # Load models
        self._load_models()

    def _load_models(self):
        """Load all required models"""
        # Load scheduler, tokenizer and models
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

        # Load base UNet
        self.unet = torch.hub.load(
            dataset='vitonhd',
            repo_or_dir='aimagelab/multimodal-garment-designer',
            source='github',
            model='mgd',
            pretrained=True
        )

        # Load temporal fine-tuned weights
        if os.path.exists(self.args.checkpoint_path):
            logger.info(f"Loading temporal weights from {self.args.checkpoint_path}")
            state_dict = torch.load(self.args.checkpoint_path, map_location='cpu')
            self.unet.load_state_dict(state_dict)
        else:
            logger.warning(f"Checkpoint not found at {self.args.checkpoint_path}, using base model")

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

    def predict_next_week(self, past_weeks_data, target_style_prompt=None):
        """
        Predict garments for next week based on past weeks

        Args:
            past_weeks_data: List of past week data
            target_style_prompt: Optional style prompt for next week
        """

        # Create pipeline
        pipe = MGDPipe(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler,
        ).to(self.device)

        pipe.enable_attention_slicing()

        predictions = []

        with torch.inference_mode():
            for week_data in past_weeks_data:
                # Use temporal conditioning from past weeks
                past_conditioning = week_data['past_conditioning']

                # Generate style prompt for next week
                if target_style_prompt is None:
                    # Use evolved style from past trends
                    base_style = past_conditioning['combined_caption_text']
                    next_week_style = f"next week trending: {base_style}"
                else:
                    next_week_style = target_style_prompt

                # Generate prediction
                generated_images = pipe(
                    prompt=[next_week_style],
                    image=week_data['image'].unsqueeze(0),
                    mask_image=week_data['inpaint_mask'].unsqueeze(0),
                    pose_map=week_data['pose_map'].unsqueeze(0),
                    sketch=past_conditioning['weighted_sketch'].unsqueeze(0),
                    height=512,
                    width=384,
                    guidance_scale=self.args.guidance_scale,
                    num_inference_steps=self.args.num_inference_steps,
                    num_images_per_prompt=1,
                    no_pose=self.args.no_pose,
                ).images

                predictions.append({
                    'generated_image': generated_images[0],
                    'style_prompt': next_week_style,
                    'base_image': week_data['im_name'],
                    'temporal_weights': week_data['temporal_weights'],
                    'category': week_data['category'],
                    'product_id': week_data['product_id']
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
            num_workers=self.args.num_workers_test,
        )

        test_dataloader = self.accelerator.prepare(test_dataloader)

        # Create output directory structure
        os.makedirs(self.args.output_dir, exist_ok=True)
        predictions_dir = os.path.join(self.args.output_dir, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)

        # Create category subdirectories
        categories = self.args.categories if self.args.categories else test_dataset.CATEGORIES
        for category in categories:
            os.makedirs(os.path.join(predictions_dir, category), exist_ok=True)

        all_predictions = []

        logger.info("Starting temporal prediction evaluation...")
        logger.info(f"Evaluating on categories: {categories}")

        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Generating predictions")):
            # Predict next week garments
            predictions = self.predict_next_week([{
                'image': batch['image'][0],
                'inpaint_mask': batch['inpaint_mask'][0],
                'pose_map': batch['pose_map'][0],
                'past_conditioning': {
                    'weighted_sketch': batch['past_conditioning']['weighted_sketch'][0],
                    'combined_caption_text': batch['past_conditioning']['combined_caption_text'][0]
                },
                'im_name': batch['im_name'][0],
                'temporal_weights': batch['temporal_weights'][0],
                'category': batch['category'][0],
                'product_id': batch['product_id'][0]
            }])

            # Save predictions
            for pred_idx, prediction in enumerate(predictions):
                category = prediction['category']
                save_name = f"{prediction['product_id']}_pred_{batch_idx:04d}.jpg"
                save_path = os.path.join(predictions_dir, category, save_name)

                prediction['generated_image'].save(save_path)

                # Store metadata
                all_predictions.append({
                    'prediction_file': os.path.join(category, save_name),
                    'style_prompt': prediction['style_prompt'],
                    'base_image': prediction['base_image'],
                    'temporal_weights': prediction['temporal_weights'].tolist(),
                    'category': category,
                    'product_id': prediction['product_id']
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

        logger.info(f"Generated {len(all_predictions)} predictions")
        logger.info(f"Category distribution: {category_stats}")
        logger.info(f"Results saved to {self.args.output_dir}")

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