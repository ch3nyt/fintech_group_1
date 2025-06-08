#!/usr/bin/env python3

import sys
sys.path.append('./src')

from datasets.temporal_vitonhd_dataset import TemporalVitonHDDataset
from transformers import CLIPTokenizer

# Test dataset loading
print("Testing TemporalVitonHDDataset loading...")

# Initialize tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Test with different parameters
dataset_path = "./dataset_vitonhd_format"

print(f"Dataset path: {dataset_path}")

# Try with minimal past weeks requirement
for num_past_weeks in [1, 2, 4]:
    print(f"\n--- Testing with num_past_weeks={num_past_weeks} ---")

    try:
        dataset = TemporalVitonHDDataset(
            dataroot_path=dataset_path,
            phase='test',
            tokenizer=tokenizer,
            num_past_weeks=num_past_weeks,
            temporal_weight_decay=0.8,
            size=(512, 384),
            category_filter=None
        )

        print(f"Dataset created successfully!")
        print(f"Number of samples: {len(dataset)}")

        if len(dataset) > 0:
            print("✅ Success! Dataset has samples.")
            # Try loading one sample
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            break
        else:
            print("⚠️ Dataset has 0 samples")

    except Exception as e:
        print(f"❌ Error creating dataset: {e}")

print("Debug completed.")