#!/usr/bin/env python3
"""
CLIP Score Evaluation Script
Evaluates CLIP-I (image-to-image) and CLIP-T (text-to-image) scores for images in a directory.
"""

import os
import json
import torch
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
import csv
import re


class CLIPEvaluator:
    """Comprehensive CLIP-based evaluator for image quality assessment"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"🔧 Initializing CLIP model on {device}...")

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

        print("✅ CLIP model loaded successfully!")

    def compute_clip_i_score(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compute CLIP image-to-image similarity score"""
        with torch.no_grad():
            inputs = self.processor(images=[image1, image2], return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features[0:1], image_features[1:2], dim=-1)
            return similarity.item()

    def compute_clip_t_score(self, image: Image.Image, text: str) -> float:
        """Compute CLIP text-to-image similarity score"""
        with torch.no_grad():
            inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(pixel_values=inputs['pixel_values'])
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
            return similarity.item()


def load_image_safely(image_path: Path) -> Optional[Image.Image]:
    """Load image with error handling"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        print(f"❌ Failed to load {image_path}: {e}")
        return None


def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get all image files from directory recursively"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))

    return sorted(image_files)


def get_fashion_text_prompts() -> List[str]:
    """Get a comprehensive list of fashion-related text prompts for evaluation"""
    return [
        "a fashionable clothing item",
        "stylish modern garment",
        "trendy fashion piece",
        "elegant clothing design",
        "contemporary fashion item",
        "sophisticated apparel",
        "high-quality clothing",
        "designer fashion piece",
        "luxury garment",
        "premium clothing item",
        "artistic fashion design",
        "beautiful clothing piece",
        "refined fashion item",
        "modern stylish clothing",
        "elegant fashion garment"
    ]


def extract_caption_from_file(captions_file: str, image_name: str) -> str:
    """Extract the Final Generation Prompt from captions file for a specific image"""
    try:
        with open(captions_file, 'r') as f:
            content = f.read()

        # Look for the image block
        pattern = f"Image: {image_name}\\.jpg.*?(?=Image: |$)"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            block = match.group(0)
            # Extract Final Generation Prompt
            prompt_match = re.search(r"Final Generation Prompt: (.*?)(?=\n[A-Z]|$)", block, re.DOTALL)
            if prompt_match:
                return prompt_match.group(1).strip()
        return None
    except Exception as e:
        print(f"Error extracting caption for {image_name}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-I and CLIP-T scores for images")

    parser.add_argument("--input_dir", type=str, required=True,
                       help="Root directory containing images to evaluate")
    parser.add_argument("--reference_dir", type=str, default=None,
                       help="Reference directory for CLIP-I comparison (optional)")
    parser.add_argument("--output_dir", type=str, default="./clip_evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--captions_file", type=str, required=True,
                       help="Path to the captions file containing Final Generation Prompts")
    parser.add_argument("--clip_i_weight", type=float, default=0.6,
                       help="Weight for CLIP-I score in combined score")
    parser.add_argument("--clip_t_weight", type=float, default=0.4,
                       help="Weight for CLIP-T score in combined score")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for processing (to manage memory)")
    parser.add_argument("--create_visualizations", action="store_true",
                       help="Create visualization plots")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], help="Device to use")

    return parser.parse_args()


def evaluate_directory(args):
    """Main evaluation function"""
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🚀 Starting CLIP evaluation on {device}")
    print(f"📁 Input directory: {args.input_dir}")
    print(f"💾 Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize evaluator
    evaluator = CLIPEvaluator(device=device)

    # Get image files
    input_dir = Path(args.input_dir)
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return

    print(f"📸 Found {len(image_files)} images to evaluate")

    # Setup reference images for CLIP-I (if provided)
    reference_images = []
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        reference_files = get_image_files(ref_dir)
        print(f"🎯 Found {len(reference_files)} reference images")

        for ref_file in reference_files[:min(100, len(reference_files))]:  # Limit to 100 refs
            ref_img = load_image_safely(ref_file)
            if ref_img:
                reference_images.append(ref_img)

    # Storage for results
    results = {
        'individual_scores': [],
        'clip_i_scores': [],
        'clip_t_scores': [],
        'combined_scores': [],
        'failed_images': []
    }

    # Process images
    print("🔄 Processing images...")
    for img_file in tqdm(image_files, desc="Evaluating images"):
        try:
            # Load image
            img = load_image_safely(img_file)
            if img is None:
                results['failed_images'].append(str(img_file))
                continue

            # Get image name without extension
            image_name = img_file.stem

            # Extract caption from captions file
            caption = extract_caption_from_file(args.captions_file, image_name)
            if caption is None:
                print(f"⚠️ No caption found for {image_name}, skipping...")
                results['failed_images'].append(str(img_file))
                continue

            # Calculate CLIP-I scores (if reference images available)
            clip_i_score = 0.0
            if reference_images:
                clip_i_scores = []
                for ref_img in reference_images:
                    score = evaluator.compute_clip_i_score(img, ref_img)
                    clip_i_scores.append(score)
                clip_i_score = np.mean(clip_i_scores)  # Average across all references

            # Calculate CLIP-T score using the extracted caption
            clip_t_score = evaluator.compute_clip_t_score(img, caption)

            # Calculate combined score
            if reference_images:
                combined_score = args.clip_i_weight * clip_i_score + args.clip_t_weight * clip_t_score
            else:
                combined_score = clip_t_score  # Only CLIP-T if no references

            # Store results
            individual_result = {
                'image_path': str(img_file.relative_to(input_dir)),
                'caption': caption,
                'clip_i_score': clip_i_score,
                'clip_t_score': clip_t_score,
                'combined_score': combined_score,
                'image_size': img.size
            }

            results['individual_scores'].append(individual_result)
            if reference_images:
                results['clip_i_scores'].append(clip_i_score)
            results['clip_t_scores'].append(clip_t_score)
            results['combined_scores'].append(combined_score)

        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
            results['failed_images'].append(str(img_file))

    # Calculate summary statistics
    summary = {
        'total_images': len(image_files),
        'processed_images': len(results['individual_scores']),
        'failed_images': len(results['failed_images']),
        'clip_t_average': np.mean(results['clip_t_scores']),
        'clip_t_std': np.std(results['clip_t_scores']),
        'clip_t_min': np.min(results['clip_t_scores']),
        'clip_t_max': np.max(results['clip_t_scores']),
        'combined_average': np.mean(results['combined_scores']),
        'combined_std': np.std(results['combined_scores']),
        'combined_min': np.min(results['combined_scores']),
        'combined_max': np.max(results['combined_scores']),
        'evaluation_config': {
            'clip_i_weight': args.clip_i_weight,
            'clip_t_weight': args.clip_t_weight,
            'reference_images_count': len(reference_images),
            'device_used': device
        }
    }

    if reference_images:
        summary.update({
            'clip_i_average': np.mean(results['clip_i_scores']),
            'clip_i_std': np.std(results['clip_i_scores']),
            'clip_i_min': np.min(results['clip_i_scores']),
            'clip_i_max': np.max(results['clip_i_scores'])
        })

    # Print summary
    print("\n" + "="*60)
    print("📊 CLIP EVALUATION RESULTS")
    print("="*60)
    print(f"📸 Total images processed: {summary['processed_images']}/{summary['total_images']}")

    if reference_images:
        print(f"🎯 CLIP-I (Image Similarity) Average: {summary['clip_i_average']:.4f} ± {summary['clip_i_std']:.4f}")
        print(f"   Range: [{summary['clip_i_min']:.4f}, {summary['clip_i_max']:.4f}]")

    print(f"📝 CLIP-T (Text Similarity) Average: {summary['clip_t_average']:.4f} ± {summary['clip_t_std']:.4f}")
    print(f"   Range: [{summary['clip_t_min']:.4f}, {summary['clip_t_max']:.4f}]")
    print(f"🎯 Combined Score Average: {summary['combined_average']:.4f} ± {summary['combined_std']:.4f}")
    print(f"   Range: [{summary['combined_min']:.4f}, {summary['combined_max']:.4f}]")

    if summary['failed_images'] > 0:
        print(f"❌ Failed images: {summary['failed_images']}")

    # Save results
    print(f"\n💾 Saving results to {output_dir}")

    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save individual scores
    with open(output_dir / "individual_scores.json", 'w') as f:
        json.dump(results['individual_scores'], f, indent=2)

    # Save as CSV for easy analysis
    with open(output_dir / "scores.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['image_path', 'caption', 'clip_t_score', 'combined_score']
        if reference_images:
            header.insert(-1, 'clip_i_score')
        writer.writerow(header)

        for item in results['individual_scores']:
            row = [item['image_path'], item['caption'], item['clip_t_score'], item['combined_score']]
            if reference_images:
                row.insert(-1, item['clip_i_score'])
            writer.writerow(row)

    # Create visualizations
    if args.create_visualizations:
        print("📈 Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('CLIP Score Distribution Analysis', fontsize=16)

        # CLIP-T histogram
        axes[0, 0].hist(results['clip_t_scores'], bins=30, alpha=0.7, color='blue')
        axes[0, 0].set_title('CLIP-T Score Distribution')
        axes[0, 0].set_xlabel('CLIP-T Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(summary['clip_t_average'], color='red', linestyle='--',
                          label=f'Mean: {summary["clip_t_average"]:.3f}')
        axes[0, 0].legend()

        # Combined score histogram
        axes[0, 1].hist(results['combined_scores'], bins=30, alpha=0.7, color='green')
        axes[0, 1].set_title('Combined Score Distribution')
        axes[0, 1].set_xlabel('Combined Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(summary['combined_average'], color='red', linestyle='--',
                          label=f'Mean: {summary["combined_average"]:.3f}')
        axes[0, 1].legend()

        # Score scatter plot
        if reference_images:
            axes[1, 0].scatter(results['clip_i_scores'], results['clip_t_scores'], alpha=0.6)
            axes[1, 0].set_title('CLIP-I vs CLIP-T Scores')
            axes[1, 0].set_xlabel('CLIP-I Score')
            axes[1, 0].set_ylabel('CLIP-T Score')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Reference Images\nCLIP-I scores not available',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('CLIP-I vs CLIP-T (N/A)')

        # Top/Bottom performers
        sorted_by_combined = sorted(results['individual_scores'],
                                   key=lambda x: x['combined_score'], reverse=True)
        top_5 = sorted_by_combined[:5]
        bottom_5 = sorted_by_combined[-5:]

        performers_text = "Top 5 Performers:\n"
        for i, item in enumerate(top_5, 1):
            performers_text += f"{i}. {Path(item['image_path']).name} ({item['combined_score']:.3f})\n"
            performers_text += f"   Caption: {item['caption'][:50]}...\n"

        performers_text += "\nBottom 5 Performers:\n"
        for i, item in enumerate(bottom_5, 1):
            performers_text += f"{i}. {Path(item['image_path']).name} ({item['combined_score']:.3f})\n"
            performers_text += f"   Caption: {item['caption'][:50]}...\n"

        axes[1, 1].text(0.05, 0.95, performers_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=8)
        axes[1, 1].set_title('Top/Bottom Performers')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "clip_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Evaluation complete! Results saved to {output_dir}")
    return summary


def main():
    args = parse_args()

    # Validate input directory
    if not Path(args.input_dir).exists():
        print(f"❌ Input directory {args.input_dir} does not exist")
        return

    # Run evaluation
    try:
        summary = evaluate_directory(args)
        print("\n🎉 CLIP evaluation completed successfully!")
    except Exception as e:
        print(f"💥 Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

