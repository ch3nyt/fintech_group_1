#!/usr/bin/env python3
"""
Example script to run DPO training for temporal fashion generation.

This script demonstrates how to use the DPO trainer with appropriate parameters
for robust temporal fashion generation training.
"""

import subprocess
import sys
import os

def run_dpo_training():
    """Run DPO training with example parameters"""

    # Define training parameters
    dataset_path = "/root/temporal_vitonhd_dataset"
    output_dir = "./temporal_vitonhd_dpo_checkpoints"

    # DPO training command
    cmd = [
        sys.executable, "train_vitonhd_dpo.py",
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--pretrained_model_name_or_path", "runwayml/stable-diffusion-inpainting",

        # Training parameters
        "--learning_rate", "1e-5",
        "--max_train_steps", "1500",
        "--batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--mixed_precision", "fp16",
        "--seed", "42",

        # DPO specific parameters
        "--num_candidates", "20",           # Generate 20 candidates per sample
        "--dpo_beta", "0.1",               # KL regularization strength
        "--dpo_weight", "0.5",             # Weight for DPO loss vs diffusion loss
        "--clip_i_weight", "0.6",          # Weight for CLIP-I (image similarity)
        "--clip_t_weight", "0.4",          # Weight for CLIP-T (text-image similarity)

        # Temporal parameters
        "--num_past_weeks", "4",
        "--temporal_weight_decay", "0.8",
        "--temporal_loss_weight", "0.3",

        # Generation parameters for DPO candidates
        "--guidance_scale", "7.5",
        "--num_inference_steps", "20",     # Faster inference for candidate generation
        "--no_pose", "True",

        # Logging and checkpointing
        "--save_steps", "250",
        "--log_steps", "50",
    ]

    print("Starting DPO training with the following parameters:")
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print("DPO Parameters:")
    print(f"  - Number of candidates per sample: 20")
    print(f"  - DPO beta (KL regularization): 0.1")
    print(f"  - DPO weight: 0.5")
    print(f"  - CLIP-I weight: 0.6")
    print(f"  - CLIP-T weight: 0.4")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run training
    try:
        print("🚀 Starting DPO training...")
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ DPO training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DPO training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
        return False

def run_dpo_training_from_checkpoint():
    """Example of resuming DPO training from a checkpoint"""

    dataset_path = "/root/temporal_vitonhd_dataset"
    output_dir = "./temporal_vitonhd_dpo_checkpoints"
    checkpoint_path = "/root/multimodal-garment-designer/temporal_vitonhd_checkpoints/temporal_vitonhd_20250531_173346/checkpoint-1000/unet.pth"

    cmd = [
        sys.executable, "train_vitonhd_dpo.py",
        "--dataset_path", dataset_path,
        "--output_dir", output_dir,
        "--resume_from_checkpoint", checkpoint_path,

        # Reduced training steps since we're fine-tuning
        "--max_train_steps", "500",
        "--learning_rate", "5e-6",  # Lower learning rate for fine-tuning

        # DPO parameters
        "--num_candidates", "15",   # Slightly fewer candidates for faster training
        "--dpo_weight", "0.7",     # Higher DPO weight for preference optimization
        "--clip_i_weight", "0.7",
        "--clip_t_weight", "0.3",

        # Other parameters
        "--batch_size", "1",
        "--gradient_accumulation_steps", "16",
        "--mixed_precision", "fp16",
        "--save_steps", "100",
        "--log_steps", "25",
    ]

    print("🔄 Starting DPO fine-tuning from checkpoint...")
    print(f"Checkpoint: {checkpoint_path}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ DPO fine-tuning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DPO fine-tuning failed with exit code {e.returncode}")
        return False

def main():
    """Main function to choose training mode"""

    print("🎨 Temporal Fashion Generation - DPO Training")
    print("=" * 50)
    print()
    print("Choose training mode:")
    print("1. Fresh DPO training")
    print("2. DPO fine-tuning from existing checkpoint")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        print("\n🆕 Starting fresh DPO training...")
        success = run_dpo_training()
    elif choice == "2":
        print("\n🔄 Starting DPO fine-tuning...")
        success = run_dpo_training_from_checkpoint()
    elif choice == "3":
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")
        return

    if success:
        print("\n🎉 Training completed successfully!")
        print("\nNext steps:")
        print("1. Check the output directory for saved checkpoints")
        print("2. Use the trained model for evaluation with eval_temporal.py")
        print("3. Compare results with and without DPO optimization")
    else:
        print("\n💡 Training tips:")
        print("- Check GPU memory usage (reduce batch_size or num_candidates if needed)")
        print("- Verify dataset path exists and contains proper data")
        print("- Monitor logs for any specific error messages")

if __name__ == "__main__":
    main()