import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
from typing import Optional, Union, List

# custom imports
from datasets.dresscode import DressCodeDataset
from datasets.vitonhd import VitonHDDataset
from mgd_pipelines.mgd_pipe import MGDPipe
from mgd_pipelines.mgd_pipe_disentangled import MGDPipeDisentangled
from utils.set_seeds import set_seed

logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for MGD")

    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Dataset parameters
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--save_steps", type=int, default=500)

    return parser.parse_args()

def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             beta: float) -> torch.FloatTensor:
    """Compute the DPO loss for a batch of policy and reference model log probabilities."""
    chosen_rewards = policy_chosen_logps
    rejected_rewards = policy_rejected_logps

    losses = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))

    return losses.mean()

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

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load UNet
    unet = torch.hub.load(dataset=args.dataset, repo_or_dir='aimagelab/multimodal-garment-designer', source='github',
                         model='mgd', pretrained=True)

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Enable memory efficient attention if requested
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    # Prepare dataset
    if args.category:
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dataset_path,
            phase='train',
            order='paired',
            radius=5,
            sketch_threshold_range=(20, 20),
            tokenizer=tokenizer,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='train',
            order='paired',
            sketch_threshold_range=(20, 20),
            radius=5,
            tokenizer=tokenizer,
            size=(512, 384),
        )
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Initialize pipeline
    pipe = MGDPipe(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
    ).to(accelerator.device)

    # Training loop
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    global_step = 0
    for step in range(args.max_train_steps):
        unet.train()

        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Generate chosen and rejected samples
                chosen_images = pipe(
                    prompt=batch["captions"],
                    image=batch["image"],
                    mask_image=batch["inpaint_mask"],
                    pose_map=batch["pose_map"],
                    sketch=batch["im_sketch"],
                    height=512,
                    width=384,
                    guidance_scale=7.5,
                    num_images_per_prompt=1,
                ).images

                # For rejected samples, we'll use a different guidance scale
                rejected_images = pipe(
                    prompt=batch["captions"],
                    image=batch["image"],
                    mask_image=batch["inpaint_mask"],
                    pose_map=batch["pose_map"],
                    sketch=batch["im_sketch"],
                    height=512,
                    width=384,
                    guidance_scale=1.0,  # Lower guidance scale for rejected samples
                    num_images_per_prompt=1,
                ).images

                # Calculate log probabilities
                chosen_logps = torch.log(torch.tensor([1.0])).to(accelerator.device)  # Placeholder
                rejected_logps = torch.log(torch.tensor([0.5])).to(accelerator.device)  # Placeholder

                # Calculate DPO loss
                loss = dpo_loss(chosen_logps, rejected_logps, args.beta)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(
        os.path.join(args.output_dir, "final_model"),
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

if __name__ == "__main__":
    main()