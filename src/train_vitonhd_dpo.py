import os
import json
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, StableDiffusionInpaintPipeline
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
import bitsandbytes as bnb

# custom imports
from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from utils.set_seeds import set_seed
from transformers import CLIPTokenizer

import torch
import torch.nn.functional as F

import traceback; traceback.print_exc()
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
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to temporal dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")

    # Temporal parameters
    parser.add_argument("--num_past_weeks", type=int, default=4, help="Number of past weeks to consider")
    parser.add_argument("--temporal_weight_decay", type=float, default=0.8, help="Weight decay for older weeks")
    parser.add_argument("--temporal_loss_weight", type=float, default=0.3, help="Weight for temporal consistency loss")

    # DPO parameters
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidate images to generate for DPO")
    parser.add_argument("--dpo_beta", type=float, default=0.1, help="KL regularization strength for DPO")
    parser.add_argument("--dpo_weight", type=float, default=0.5, help="Weight for DPO loss")
    parser.add_argument("--clip_i_weight", type=float, default=0.6, help="Weight for CLIP-I score")
    parser.add_argument("--clip_t_weight", type=float, default=0.4, help="Weight for CLIP-T score")
    parser.add_argument("--dpo_frequency", type=float, default=0.01, help="Frequency of applying DPO loss (0.05 = 5% of batches)")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps for candidate generation")
    parser.add_argument("--image_size", type=tuple, default=(432, 288), help="Image size (height, width) for training and generation")

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
    parser.add_argument("--project_name", type=str, default="temporal-sd2-inpainting-dpo", help="Wandb project name")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    

    return parser.parse_args()


def generate_candidates(
    pipe: StableDiffusionInpaintPipeline,
    batch: Dict,
    num_candidates: int,
    num_inference_steps: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Generate multiple candidate images for DPO using SD2 inpainting pipeline"""
    args= parse_args()
    candidates = []

    # Memory optimization: Clear cache before generation
    torch.cuda.empty_cache()

    # Reduce number of candidates if we're getting OOM
    actual_num_candidates = min(num_candidates, 8)  # Limit to max 8 candidates

    try:
        # Extract inputs from batch - use tensors directly like in eval_temporal.py
        images = batch['image']
        masks = batch['inpaint_mask']

        # Handle batched tensors by taking the first item
        if isinstance(images, torch.Tensor) and images.ndim > 3:
            image = images[0]
        else:
            image = images[0] if isinstance(images, (list, tuple)) else images

        if isinstance(masks, torch.Tensor) and masks.ndim > 3:
            mask_image = masks[0]
        else:
            mask_image = masks[0] if isinstance(masks, (list, tuple)) else masks

        # Get prompt - handle both string and token formats
        # 所以這裡其實只要 string 的 caption? 在 main 將 batch['captions'] 處理成 List[str] 即可
        if isinstance(batch['captions'][0], str):
            prompt = batch['captions'][0]
        else:
            prompt = "fashionable clothing item"  # default prompt

        # Clean up prompt to avoid repetition issues
        prompt = prompt.strip()
        if len(prompt) > 100:  # Truncate very long prompts
            prompt = prompt[:100]

        # Convert tensors to PIL Images for SD2 inpainting pipeline
        image_pil = transforms.ToPILImage()(((image + 1) / 2).clamp(0, 1).cpu())

        # Convert mask tensor to PIL
        if mask_image.dim() == 3 and mask_image.shape[0] == 1:
            mask_tensor = mask_image.squeeze(0)
        else:
            mask_tensor = mask_image
        mask_pil = transforms.ToPILImage()(mask_tensor.cpu())

        # Memory optimization: Use smaller resolution for candidate generation
        target_height, target_width = 144, 96  # Ultra-small resolution for maximum memory efficiency

        # Generate candidates with different seeds for diversity
        for i in range(actual_num_candidates):
            try:
                # Memory optimization: Clear cache before each generation
                if i > 0:
                    torch.cuda.empty_cache()

                # Use different seeds for diversity
                generator = torch.Generator(device=device).manual_seed(42 + i)

                # Call SD2 inpainting pipeline with memory optimizations
                with torch.no_grad():
                    # Enable memory efficient settings
                    # pipe.enable_attention_slicing() (外層的 sd2_pipe 已經啟用了)
                    # Remove CPU offloading as it conflicts with training

                    result = pipe(
                        prompt=prompt,
                        image=image_pil,
                        mask_image=mask_pil,
                        height=target_height,  # Reduced resolution
                        width=target_width,   # Reduced resolution
                        guidance_scale=5.0,   # Reduced from 7.5 for memory
                        num_inference_steps=max(10, num_inference_steps // 2),  # Reduced steps
                        strength=0.99,
                        generator=generator
                    )

                # Extract the generated image
                if hasattr(result, 'images') and len(result.images) > 0:
                    generated_img = result.images[0]

                    # Convert PIL Image to tensor for consistency
                    if hasattr(generated_img, 'convert'):  # It's a PIL Image
                        # Resize back to original size if needed
                        if target_height != args.image_size[0] or target_width != args.image_size[1]:
                            generated_img = generated_img.resize(args.image_size, Image.Resampling.LANCZOS)

                        img_tensor = transforms.ToTensor()(generated_img.convert('RGB'))
                        # Normalize to [-1, 1] to match training data format
                        img_tensor = img_tensor * 2.0 - 1.0
                        candidates.append(img_tensor)
                    else:
                        # Already a tensor
                        candidates.append(generated_img)
                else:
                    print(f"No images generated for candidate {i}")

            except Exception as e:
                print(f"Failed to generate candidate {i}: {e}")
                # Clear cache on error and continue
                torch.cuda.empty_cache()
                continue

    except Exception as e:
        print(f"Error in generate_candidates: {e}")
        return []

    # Final cache clear
    torch.cuda.empty_cache()

    print(f"Successfully generated {len(candidates)} candidates using SD2 inpainting")
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

'''
10/07 替換成真正的 DPO

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
    # 10/06 不應該跟 0 比? 應該要與真實 noise 比較
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
'''
# ===== DPO helpers =====

@torch.no_grad()
def _encode_to_latents(x_img_bchw: torch.Tensor, vae, scaling: float, use_mean: bool = True) -> torch.Tensor:
    """
    x_img_bchw: 影像張量，範圍須為 [-1,1]，shape [B,3,H,W]，dtype/裝置需與 VAE 相容
    回傳：latent，shape [B,4,H/8,W/8]，已乘 scaling_factor
    """
    enc = vae.encode(x_img_bchw)
    lat = enc.latent_dist.mean if use_mean else enc.latent_dist.sample()
    return lat * scaling

def _masked_mse_per_sample(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    逐樣本 MSE（可選用 mask 做區域平均）。回傳 [B]
    pred/target: [B,C,H,W]；mask: [B,1,H,W] 或 [B,H,W]，1=計分
    """
    mse = (pred - target) ** 2
    if mask is not None:
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(1)  # -> [B,1,H,W]
        mask = mask.to(pred.dtype)
        mse = mse * mask
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        return mse.flatten(1).sum(dim=1) / denom
    else:
        return mse.flatten(1).mean(dim=1)

def compute_true_dpo_loss(
    model_pref: torch.Tensor, model_rej: torch.Tensor,
    ref_pref: torch.Tensor,   ref_rej: torch.Tensor,
    target_pref: torch.Tensor, target_rej: torch.Tensor,
    mask_latent: torch.Tensor | None,
    beta: float
) -> torch.Tensor:
    """
    真正的 DPO：log p 以 -MSE 近似，且 y+ / y- 用相同 (t, ε) 與相同條件。
    所有張量 shape 應為 [B,C,H,W]（除了 mask 可為 [B,1,H,W]）。
    """
    ref_pref = ref_pref.detach()
    ref_rej  = ref_rej.detach()

    # log p ≈ -MSE
    mse_m_p = _masked_mse_per_sample(model_pref, target_pref, mask_latent)  # [B]
    mse_m_r = _masked_mse_per_sample(model_rej,  target_rej,  mask_latent)
    mse_r_p = _masked_mse_per_sample(ref_pref,   target_pref, mask_latent)
    mse_r_r = _masked_mse_per_sample(ref_rej,    target_rej,  mask_latent)

    # Δ_model = logp(y+) - logp(y-) ≈ -(MSE+ - MSE-) = MSE- - MSE+
    delta_model = (mse_m_r - mse_m_p)  # [B]
    delta_ref   = (mse_r_r - mse_r_p)  # [B]

    # DPO = -log σ(β(Δ_model - Δ_ref))
    return -F.logsigmoid(beta * (delta_model - delta_ref)).mean()

def true_dpo_forward_once(
    y_pos_chw: torch.Tensor,          # 候選最佳 y+，[-1,1]，[3,H,W]
    y_neg_chw: torch.Tensor,          # 候選最差 y-，[-1,1]，[3,H,W]
    img0_chw: torch.Tensor,           # 原始條件中的 image（同一筆樣本），[-1,1]，[3,H,W]
    mask0_chw: torch.Tensor,          # 對應的 inpaint mask，1=要修補，[1,H,W]
    enc_hidden_states_bld: torch.Tensor,  # [1, seq_len, D]（只給這一筆樣本）
    unet, ref_unet, vae, noise_scheduler,
    vae_scaling: float,
    beta: float,
    device: torch.device
) -> torch.Tensor:
    """
    以單一 pair（取 batch 的第 0 筆）計算真正 DPO 損失。
    流程：encode y+/y- -> 同一 (t,ε) 造噪 -> 組 9-ch inpaint 輸入 -> model/ref forward -> DPO。
    """
    # --- 尺寸/型別對齊 ---
    H, W = img0_chw.shape[-2:]
    def _resize_img_chw(x):
        x = x.unsqueeze(0) if x.dim()==3 else x  # -> [1,3,H,W]
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    y_pos = _resize_img_chw(y_pos_chw.clamp(-1,1)).to(device=device, dtype=vae.dtype)   # [1,3,H,W]
    y_neg = _resize_img_chw(y_neg_chw.clamp(-1,1)).to(device=device, dtype=vae.dtype)   # [1,3,H,W]
    img0  = img0_chw.unsqueeze(0).to(device=device, dtype=vae.dtype)                    # [1,3,H,W]
    mask0 = mask0_chw.unsqueeze(0).to(device=device)                                    # [1,1,H,W]

    # --- encode 到 latent（用 posterior mean 比較穩） ---
    with torch.no_grad():
        lat_pos = _encode_to_latents(y_pos, vae, vae_scaling, use_mean=True)   # [1,4,h,w]
        lat_neg = _encode_to_latents(y_neg, vae, vae_scaling, use_mean=True)
        masked_img = img0 * (1 - mask0)                                       # [1,3,H,W]
        masked_lat = _encode_to_latents(masked_img, vae, vae_scaling, use_mean=True)
        mask_lat  = F.interpolate(mask0.float(), size=lat_pos.shape[-2:], mode='nearest')  # [1,1,h,w]

    # --- 抽同一組 (t, ε) 並加噪 ---
    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
    eps = torch.randn_like(lat_pos)
    x_t_pos = noise_scheduler.add_noise(lat_pos, eps, t)
    x_t_neg = noise_scheduler.add_noise(lat_neg, eps, t)

    # --- 組 SD2 Inpainting 的 9 通道輸入：4 + 1 + 4 ---
    inp_pos = torch.cat([x_t_pos, mask_lat, masked_lat], dim=1)  # [1,9,h,w]
    inp_neg = torch.cat([x_t_neg, mask_lat, masked_lat], dim=1)

    # --- 目標 target（依 scheduler.prediction_type）---
    pred_type = getattr(noise_scheduler.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        tgt_pos = eps
        tgt_neg = eps
    elif pred_type == "v_prediction":
        tgt_pos = noise_scheduler.get_velocity(lat_pos, eps, t)
        tgt_neg = noise_scheduler.get_velocity(lat_neg, eps, t)
    else:
        raise ValueError(f"Unknown prediction_type: {pred_type}")

    # --- 前向：model / ref ---
    model_pos = unet(inp_pos, t, encoder_hidden_states=enc_hidden_states_bld).sample
    model_neg = unet(inp_neg, t, encoder_hidden_states=enc_hidden_states_bld).sample
    with torch.no_grad():
        ref_pos = ref_unet(inp_pos, t, encoder_hidden_states=enc_hidden_states_bld).sample
        ref_neg = ref_unet(inp_neg, t, encoder_hidden_states=enc_hidden_states_bld).sample

    # --- DPO（只在 inpaint 區域計分）---
    dpo_loss = compute_true_dpo_loss(
        model_pos, model_neg, ref_pos, ref_neg,
        tgt_pos, tgt_neg, mask_latent=mask_lat, beta=beta
    )
    return dpo_loss
# ===== end DPO helpers =====

def _masked_mse_per_sample(pred: torch.Tensor,
                           target: torch.Tensor,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    回傳 shape = [B] 的逐樣本 MSE。
    pred/target: [B,C,H,W]
    mask:        [B,1,H,W] 或 [B,H,W] 或 [B]；1=計分區域，0=忽略
    """
    mse = (pred - target) ** 2
    if mask is not None:
        # 將 mask broadcast 到 pred 形狀
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(1)             # -> [B,1,H,W]
        mask = mask.to(pred.dtype)
        mse = mse * mask
        # 逐樣本平均：用有效像素數做歸一化，避免全零導致 NaN
        denom = mask.flatten(1).sum(dim=1).clamp_min(1.0)
        mse = mse.flatten(1).sum(dim=1) / denom
        return mse  # [B]
    else:
        # 無 mask 時，對空間與通道取平均，保留 batch 維
        return F.mse_loss(pred, target, reduction='none').flatten(1).mean(dim=1)

'''
def compute_dpo_loss(
    model_preferred_logits: torch.Tensor,   # UNet 對 y⁺ 的預測 (ε̂ 或 v̂)，shape [B,C,H,W]
    model_rejected_logits: torch.Tensor,    # UNet 對 y⁻ 的預測
    ref_preferred_logits: torch.Tensor,     # 參考模型對 y⁺ 的預測
    ref_rejected_logits: torch.Tensor,      # 參考模型對 y⁻ 的預測
    target_preferred: torch.Tensor,         # 真實 target：若 pred_type="epsilon" 則是真實 ε；若 "v_prediction" 則為 v
    target_rejected: torch.Tensor,          # 與上同理
    mask_latent: torch.Tensor | None = None,# 可選，inpainting 修補區域的 latent 級遮罩，1=計分
    beta: float = 0.1
) -> torch.Tensor:
    """
    真正的 DPO：以 -MSE 當作 log p 的 proxy。
    需保證 y⁺/y⁻ 使用「相同的 (t, ε)」與相同的條件（文字/遮罩/被遮影像）來評估。
    """

    # 參考模型不反傳梯度（再次保險）
    ref_preferred_logits = ref_preferred_logits.detach()
    ref_rejected_logits  = ref_rejected_logits.detach()

    # 用 masked/非 masked MSE 做為 -loglik 的相反數
    # logprob ≈ -MSE(pred, target)
    model_mse_pref = _masked_mse_per_sample(model_preferred_logits, target_preferred, mask_latent)  # [B]
    model_mse_rej  = _masked_mse_per_sample(model_rejected_logits,  target_rejected,  mask_latent)  # [B]
    ref_mse_pref   = _masked_mse_per_sample(ref_preferred_logits,   target_preferred, mask_latent)  # [B]
    ref_mse_rej    = _masked_mse_per_sample(ref_rejected_logits,    target_rejected,  mask_latent)  # [B]

    # log p ≈ -MSE，因此 Δ_model = logp(y⁺) - logp(y⁻) ≈ -(MSE⁺ - MSE⁻) = (MSE⁻ - MSE⁺)
    delta_model = (model_mse_rej - model_mse_pref)  # [B]
    delta_ref   = (ref_mse_rej  - ref_mse_pref)     # [B]

    # DPO loss: -log σ(β(Δ_model - Δ_ref))
    dpo = -F.logsigmoid(beta * (delta_model - delta_ref))  # [B]
    return dpo.mean()
'''

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
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
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

    # Set PyTorch CUDA memory allocation to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    if args.resume_from_checkpoint and args.resume_from_checkpoint.strip():
        # Validate checkpoint path exists
        if not os.path.exists(args.resume_from_checkpoint):
            logger.error(f"ERROR: Checkpoint file not found: {args.resume_from_checkpoint}")
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_from_checkpoint}")

        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        try:
            state_dict = torch.load(args.resume_from_checkpoint, map_location="cpu", weights_only=True)
            unet.load_state_dict(state_dict)
            logger.info(f"Successfully loaded checkpoint from {args.resume_from_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise e
    else:
        # Load SD2 inpainting UNet directly (matches eval_temporal.py)
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet"
        )
        logger.info(f"Successfully loaded SD2 inpainting UNet model from {args.pretrained_model_name_or_path}")
        logger.info(f"UNet input channels: {unet.config.in_channels}")
        if args.resume_from_checkpoint:
            logger.warning(f"Resume checkpoint path provided but empty or whitespace: '{args.resume_from_checkpoint}'. Starting fresh training.")

    # Create reference model for DPO (frozen copy)
    ref_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
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

    # Memory optimizations for VAE
    try:
        vae.enable_slicing()
    except Exception as e:
        logger.warning(f"Could not enable VAE slicing: {e}")

    try:
        if hasattr(vae, 'enable_tiling'):
            vae.enable_tiling()
    except Exception as e:
        logger.warning(f"Could not enable VAE tiling: {e}")

    # Prepare dataset
    # 10/20 改成 576 for 原本的圖片是 1166*1750 (2:3 直式)
    try:
        train_dataset = TemporalVitonHDDataset(
            dataroot_path=args.dataset_path,
            phase='train',
            tokenizer=tokenizer,
            num_past_weeks=args.num_past_weeks,
            temporal_weight_decay=args.temporal_weight_decay,
            size=args.image_size,
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
    # 為了節省記憶體再換更 efficient 的
    '''
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    '''
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(), lr=5e-6, weight_decay=1e-2
    )

    # Prepare everything with accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move models to device
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    ref_unet.to('cpu')

    # Create SD2 inpainting pipeline for candidate generation
    sd2_pipe = StableDiffusionInpaintPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        scheduler=noise_scheduler,
        tokenizer=tokenizer,
        safety_checker=None,  # Disable safety checker for speed
        feature_extractor=None,
    ).to(accelerator.device)

    # Memory optimizations for the pipeline (safe for training)
    sd2_pipe.enable_attention_slicing()

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
    
    '''
    def to_dev(x, device=None, fp_dtype=None):
        """
        安全地把張量搬到 device，僅在浮點類型時轉成指定 dtype。
        - device: 預設 accelerator.device
        - fp_dtype: 想用的浮點 dtype（例如 unet.dtype 或 weight_dtype）
        """
        if not isinstance(x, torch.Tensor):
            return x
        device = device or accelerator.device
        if x.dtype.is_floating_point:          # 只轉浮點
            return x.to(device=device, dtype=(fp_dtype or x.dtype))
        else:
            return x.to(device=device)          # int/bool 不轉 dtype
    '''
    
    for epoch in range(1000):  # Large number, will break with max_train_steps
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Extract batch data
                images = batch['image']
                masks = batch['inpaint_mask']
                captions = batch['captions']
                past_conditioning = batch['past_conditioning']
                temporal_weights = batch['temporal_weights']

                batch_size = images.shape[0]

                # Encode images to latent space
                latents = vae.encode(images).latent_dist.sample()
                vae_scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
                latents = latents * vae_scaling_factor

                # 10/06 將這些拉入 cpu 內?
                unet.train()                  # from_pretrained 預設 eval，記得開訓練模式
                vae.eval(); text_encoder.eval(); ref_unet.eval()

                device = accelerator.device
                images = images.to(device=device, dtype=vae.dtype)
                masks  = masks.to(device=device, dtype=images.dtype)
                temporal_weights = temporal_weights.to(device)

                # 若 past_conditioning 是 dict，遞迴搬 tensor
                def _to_dev(x):
                    if torch.is_tensor(x): return x.to(device)
                    if isinstance(x, dict): return {k:_to_dev(v) for k,v in x.items()}
                    if isinstance(x, (list, tuple)): return type(x)(_to_dev(v) for v in x)
                    return x
                past_conditioning = _to_dev(past_conditioning)



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
                '''
                原本這樣寫
                encoder_hidden_states = text_encoder(captions)[0]
                10/06 先丟 tokenizer 再去 text_encoder
                '''
                def normalize_to_text_list(x):
                    # 期望：x 最終變成 List[str]
                    if x is None:
                        return [""]
                    # dataloader 取回來常是 list/tuple/ndarray
                    if isinstance(x, np.ndarray):
                        x = x.tolist()
                    if isinstance(x, (list, tuple)):
                        return ["" if t is None else str(t) for t in x]
                    # 單一元素（純字串或數值）
                    return ["" if x is None else str(x)]

                batch["captions"] = normalize_to_text_list(batch["captions"])  # → List[str]

                tok = tokenizer(
                    batch["captions"],
                    padding="max_length",                     # 對齊到 model_max_length（SD/CLIP 常見 77）
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                tok = {k: v.to(accelerator.device) for k, v in tok.items()}

                # 你原註解寫「文字編碼固定」→ no_grad 是合理的
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(**tok).last_hidden_state   # [B, 77, D]

                # Prepare SD2 inpainting conditioning (9 channels total)
                # Resize mask to latent space
                masks_resized = F.interpolate(
                    masks, size=(latents.shape[2], latents.shape[3]), mode='nearest'
                )

                # Encode masked images
                masked_images = images * (1 - masks)
                masked_image_latents = vae.encode(masked_images).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae_scaling_factor

                # SD2 inpainting concatenation: [noisy_latents (4ch), mask (1ch), masked_image_latents (4ch)] = 9ch
                unet_input = torch.cat([
                    noisy_latents,         # 4 channels
                    masks_resized,         # 1 channel
                    masked_image_latents   # 4 channels
                ], dim=1)  # Total: 9 channels

                # Predict noise with SD2 inpainting UNet
                model_pred = unet(
                    unet_input,
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

                # (重點在這裡)DPO loss computation (with probability) - only on first accumulation step
                dpo_loss = torch.tensor(0.0).to(accelerator.device)
                if (random.random() < args.dpo_frequency and global_step > 0 and
                    last_dpo_step != global_step):  # Only once per global step
                    try:
                        last_dpo_step = global_step  # Mark this step as having DPO
                        print(f"\n🔥 ATTEMPTING DPO at step {global_step} (once per step)")

                        # Memory optimization: Clear cache before DPO
                        torch.cuda.empty_cache()

                        # Step 1: Generate candidates WITHOUT gradients for scoring
                        with torch.no_grad():
                            candidates = generate_candidates(
                                sd2_pipe, batch, args.num_candidates,
                                args.num_inference_steps, accelerator.device
                            )

                        # Only proceed if we have candidates
                        if len(candidates) >= 2:  # Need at least 2 candidates for preference pairs
                            print(f"✅ Generated {len(candidates)} candidates for DPO")

                            # Step 2: Score candidates using CLIP (no gradients needed)
                            clip_scores = []
                            with torch.no_grad():
                                for idx, candidate in enumerate(candidates):
                                    candidate = (candidate.clamp(-1,1) + 1) / 2  # [-1,1] -> [0,1]
                                    candidate = candidate.to(accelerator.device)

                                    # Prepare images for CLIP processing
                                    candidate_pil = transforms.ToPILImage()(candidate.cpu())
                                    target_pil = transforms.ToPILImage()(((images[0].detach().clamp(-1,1) + 1) / 2).cpu())

                                    # CLIP-I score (image-image similarity)
                                    target_inputs = clip_scorer.processor(images=target_pil, return_tensors="pt")
                                    candidate_inputs = clip_scorer.processor(images=candidate_pil, return_tensors="pt")

                                    target_features = clip_scorer.model.get_image_features(target_inputs['pixel_values'].to(accelerator.device))
                                    candidate_features = clip_scorer.model.get_image_features(candidate_inputs['pixel_values'].to(accelerator.device))
                                    target_features = F.normalize(target_features, dim=-1)  # 10/06 正規化
                                    candidate_features = F.normalize(candidate_features, dim=-1)
                                    clip_i_score = F.cosine_similarity(target_features, candidate_features, dim=-1)

                                    # CLIP-T score (text-image similarity)
                                    caption_text = batch["captions"][0] if batch["captions"][0] != "" else "A person wearing clothing"
                                    text_inputs = clip_scorer.processor(text=caption_text, return_tensors="pt", padding=True, truncation=True)
                                    text_features = clip_scorer.model.get_text_features(text_inputs['input_ids'].to(accelerator.device))
                                    clip_t_score = F.cosine_similarity(text_features, candidate_features, dim=-1)

                                    # Combined score
                                    combined_score = args.clip_i_weight * clip_i_score + args.clip_t_weight * clip_t_score
                                    clip_scores.append(combined_score.item())

                                    # Debug: Print individual scores
                                    print(f"  Candidate {idx}: CLIP-I={clip_i_score.item():.4f}, CLIP-T={clip_t_score.item():.4f}, "
                                          f"Combined={combined_score.item():.4f} (weights: I={args.clip_i_weight}, T={args.clip_t_weight})")

                                    # Memory cleanup after each candidate
                                    del candidate, target_features, candidate_features, text_features
                                    torch.cuda.empty_cache()

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
                                    # Step 4: ULTRA-AGGRESSIVE memory-efficient DPO
                                    print(f"🔧 Using ULTRA-AGGRESSIVE DPO (no UNet forward passes)...")

                                    # Instead of running UNet forward passes (which cause OOM),
                                    # use a simpler preference-based loss on the current predictions

                                    # Use the existing model predictions from the main training loop
                                    # Apply a simple preference learning signal based on CLIP scores

                                    try:
                                        # Compute score-based preference signal
                                        score_ratio = best_score / worst_score if worst_score > 0 else 1.0
                                        preference_strength = min(max(score_ratio - 1.0, 0.0), 1.0)  # Clamp to [0, 1]

                                        # Simple DPO-inspired loss: encourage better predictions
                                        # Use the current model prediction and add a small preference penalty
                                        '''10/07 想替換成真正的 DPO'''
                                        # 如果用真正的 dpo_loss = preference_strength * 0.1 * F.mse_loss(model_pred, target)
                                        # ==== 真正 DPO：用 best vs. worst pair 做一次 forward ====
                                        try:
                                            # 取最優與最差候選（你上面已經算過 best_idx / worst_idx）
                                            y_pos = candidates[best_idx]  # [-1,1], [3,h,w] (可能在 CPU)
                                            y_neg = candidates[worst_idx]

                                            # 檢查分數差距；太小就跳過（避免噪聲偏好對）
                                            if abs(score_diff) <= 1e-4:
                                                print(f"❌ Score difference too small ({score_diff:.6f} ≤ 1e-4), skipping true DPO")
                                                dpo_loss = torch.tensor(0.0, device=accelerator.device)
                                            else:
                                                # 資料與條件：只用 batch 的第 0 筆（你目前候選也是用 images[0], masks[0]）
                                                img0   = images[0]             # [-1,1], [3,H,W], 已在 device
                                                mask0  = masks[0]              # [1,H,W]
                                                enc_hs = encoder_hidden_states[0:1]  # [1, seq_len, D]

                                                # 將候選搬到同裝置/型別（保持 [-1,1]）
                                                y_pos = y_pos.to(device=accelerator.device, dtype=vae.dtype)
                                                y_neg = y_neg.to(device=accelerator.device, dtype=vae.dtype)
                                                ref_unet.to(accelerator.device)  # 確保 ref_unet 在 GPU     
                                                # 真正 DPO 前向一次（單 pair）
                                                dpo_loss = true_dpo_forward_once(
                                                    y_pos_chw=y_pos, y_neg_chw=y_neg,
                                                    img0_chw=img0, mask0_chw=mask0,
                                                    enc_hidden_states_bld=enc_hs,
                                                    unet=accelerator.unwrap_model(unet),   # 建議用 unwrap_model，與 pipe 相容
                                                    ref_unet=ref_unet,
                                                    vae=vae, noise_scheduler=noise_scheduler,
                                                    vae_scaling=vae_scaling_factor,
                                                    beta=args.dpo_beta,
                                                    device=accelerator.device
                                                )

                                                print(f"✅ TRUE DPO LOSS: {dpo_loss.item():.6f} (beta={args.dpo_beta})")
                                                ref_unet.to('cpu')  # 移回 CPU 省記憶體

                                        except Exception as dpo_error:
                                            print(f"❌ TRUE DPO failed: {dpo_error}")
                                            dpo_loss = torch.tensor(0.0, device=accelerator.device)
                                        # ==== end 真正 DPO ====
                                        # print(f"✅ SIMPLIFIED DPO LOSS: {dpo_loss.item():.4f}")
                                        # print(f"   Score ratio: {score_ratio:.4f}")
                                        # print(f"   Preference strength: {preference_strength:.4f}")
                                        # print(f"   Best score: {best_score:.4f}, Worst score: {worst_score:.4f}")

                                    except Exception as dpo_error:
                                        print(f"❌ Simplified DPO failed: {dpo_error}")
                                        dpo_loss = torch.tensor(0.0).to(accelerator.device)

                                else:
                                    print(f"❌ Score difference too small ({score_diff:.6f} < 0.0001), skipping DPO")
                                    dpo_loss = torch.tensor(0.0).to(accelerator.device)
                            else:
                                print(f"❌ Not enough CLIP scores ({len(clip_scores)}), skipping DPO")
                                dpo_loss = torch.tensor(0.0).to(accelerator.device)
                        else:
                            print(f"❌ Not enough candidates generated ({len(candidates)}), skipping DPO loss")
                            dpo_loss = torch.tensor(0.0).to(accelerator.device)

                        # Memory cleanup after DPO
                        if 'candidates' in locals():
                            del candidates
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"💥 DPO computation failed: {e}")
                        traceback.print_exc()
                        dpo_loss = torch.tensor(0.0).to(accelerator.device)

                        # Ensure ref_unet is back on GPU even if there's an error
                        # ref_unet.to(accelerator.device)
                        # torch.cuda.empty_cache()

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

                    # Periodic memory cleanup
                    if global_step % 100 == 0:
                        torch.cuda.empty_cache()

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