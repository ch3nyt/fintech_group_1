import json
import os
import pathlib
import random
import sys
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from collections import defaultdict
import re

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))


class TemporalVitonHDDataset(data.Dataset):
    """
    Dataset for temporal garment prediction with VITONhd format
    Handles 6 categories with weekly organization
    """

    CATEGORIES = ['top5acc', 'top5gfb', 'top5glb', 'top5gub', 'top5shoe', 'top5underwear']

    def __init__(
        self,
        dataroot_path: str,
        phase: str,  # 'train', 'val', or 'test'
        tokenizer,
        num_past_weeks: int = 4,
        temporal_weight_decay: float = 0.8,
        sketch_threshold_range: Tuple[int, int] = (20, 127),
        size: Tuple[int, int] = (432, 288),
        category_filter: Optional[List[str]] = None,  # Filter specific categories
    ):
        super(TemporalVitonHDDataset, self).__init__()

        self.dataroot = pathlib.Path(dataroot_path)
        self.phase = phase
        self.num_past_weeks = num_past_weeks
        self.temporal_weight_decay = temporal_weight_decay
        self.sketch_threshold_range = sketch_threshold_range
        self.height = size[0]
        self.width = size[1]
        self.tokenizer = tokenizer

        # Categories to use
        self.categories = category_filter if category_filter else self.CATEGORIES

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load captions
        self.captions_dict = self._load_captions()

        # Build dataset structure
        self.weekly_data = self._build_weekly_structure()
        self.samples = self._create_temporal_samples()

        print(f"Created {self.__class__.__name__} with {len(self.samples)} samples")
        print(f"Categories: {self.categories}")
        print(f"Weeks found: {len(self.weekly_data)} unique weeks")

    def _load_captions(self) -> Dict[str, List[str]]:
        """Load captions from the single captions.json file"""
        caption_file = self.dataroot / 'captions.json'
        if not caption_file.exists():
            print(f"Warning: captions.json not found at {caption_file}")
            return {}

        with open(caption_file, 'r') as f:
            captions = json.load(f)

        print(f"Loaded {len(captions)} product captions")
        return captions

    def _parse_week_folder(self, week_folder: str) -> Tuple[int, int]:
        """Parse year and week number from folder name like '2020-week4'"""
        match = re.match(r'(\d{4})-week(\d+)', week_folder)
        if match:
            year = int(match.group(1))
            week = int(match.group(2))
            return year, week
        return 0, 0

    def _build_weekly_structure(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Build structure: {week_key: {category: [items]}}
        where week_key is like '2020-04' for sorting
        """
        weekly_data = defaultdict(lambda: defaultdict(list))

        # Process each category
        for category in self.categories:
            images_path = self.dataroot / self.phase / 'images' / category

            if not images_path.exists():
                print(f"Warning: Category path not found: {images_path}")
                continue

            # Process each week folder in this category
            for week_folder in sorted(images_path.iterdir()):
                if not week_folder.is_dir():
                    continue

                year, week_num = self._parse_week_folder(week_folder.name)
                if year == 0:
                    continue

                # Create sortable week key
                week_key = f"{year:04d}-{week_num:02d}"

                # Process images in this week
                for img_file in week_folder.glob('*.jpg'):
                    product_id = img_file.stem  # e.g., '0647982001'

                    # Check if we have captions for this product
                    if product_id not in self.captions_dict:
                        continue

                    item_data = {
                        'image_path': str(img_file.relative_to(self.dataroot)),
                        'product_id': product_id,
                        'category': category,
                        'week_folder': week_folder.name,
                        'week_key': week_key,
                        'year': year,
                        'week_num': week_num
                    }

                    weekly_data[week_key][category].append(item_data)

        # Convert to regular dict and sort by week
        sorted_weekly_data = {}
        for week_key in sorted(weekly_data.keys()):
            sorted_weekly_data[week_key] = dict(weekly_data[week_key])

        return sorted_weekly_data

    def _create_temporal_samples(self) -> List[Dict]:
        """Create training samples with temporal context

        BUGFIX: Fixed the issue where all items from the same category were getting
        the same "next week" text due to random.choice() being deterministic across
        items in the same category. Now uses hash-based selection for diversity.
        """
        samples = []
        week_keys = list(self.weekly_data.keys())

        # Need at least num_past_weeks + 1 weeks
        if len(week_keys) <= self.num_past_weeks:
            print(f"Warning: Not enough weeks ({len(week_keys)}) for temporal window ({self.num_past_weeks})")
            return samples

        # Create samples for each valid week window
        for i in range(self.num_past_weeks, len(week_keys)):
            target_week = week_keys[i]
            past_weeks = week_keys[max(0, i-self.num_past_weeks):i]

            # Check if there's a next week for actual text comparison
            next_week = week_keys[i + 1] if i + 1 < len(week_keys) else None

            # Calculate temporal weights
            weights = [self.temporal_weight_decay ** (len(past_weeks) - j - 1)
                      for j in range(len(past_weeks))]
            weights = np.array(weights) / sum(weights)  # Normalize

            # For each category in target week
            for category, items in self.weekly_data[target_week].items():
                for target_item in items:
                    # Check if we have data in past weeks for this category
                    past_weeks_with_data = [w for w in past_weeks
                                          if category in self.weekly_data[w]
                                          and len(self.weekly_data[w][category]) > 0]

                    # Get next week's actual data for comparison (if available)
                    next_week_item = None
                    if next_week and category in self.weekly_data[next_week] and len(self.weekly_data[next_week][category]) > 0:
                        # FIXED: Use deterministic selection based on target item to avoid same text for all items
                        # This ensures each target item gets a consistent but different next week item
                        next_week_items = self.weekly_data[next_week][category]
                        item_hash = hash(target_item['product_id']) % len(next_week_items)
                        next_week_item = next_week_items[item_hash]

                    if len(past_weeks_with_data) > 0:
                        samples.append({
                            'target_item': target_item,
                            'target_week': target_week,
                            'past_weeks': past_weeks_with_data,
                            'temporal_weights': weights[-len(past_weeks_with_data):],
                            'category': category,
                            'next_week_item': next_week_item,  # For actual next week text
                            'next_week': next_week
                        })

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        full_path = self.dataroot / image_path
        image = Image.open(full_path).convert('RGB')
        image = image.resize((self.width, self.height))
        return self.transform(image)

    def _load_sketch(self, image_path: str) -> torch.Tensor:
        """Load sketch corresponding to image"""
        # Replace 'images' with 'im_sketch' in path
        sketch_path = image_path.replace('/images/', '/im_sketch/').replace('.jpg', '.png')
        full_path = self.dataroot / sketch_path

        if not full_path.exists():
            # Generate dummy sketch if not found
            return torch.zeros(1, self.height, self.width)

        sketch_threshold = random.randint(*self.sketch_threshold_range)
        sketch = Image.open(full_path).convert('L')
        sketch = sketch.resize((self.width, self.height))
        sketch = ImageOps.invert(sketch)
        sketch = sketch.point(lambda p: 255 if p > sketch_threshold else 0)
        sketch = transforms.functional.to_tensor(sketch)
        return 1 - sketch

    def _load_segmentation(self, image_path: str) -> torch.Tensor:
        """Load segmentation mask"""
        # Replace 'images' with 'im_seg' in path
        seg_path = image_path.replace('/images/', '/im_seg/').replace('.jpg', '.png')
        full_path = self.dataroot / seg_path

        if not full_path.exists():
            # Generate dummy segmentation if not found
            seg = torch.zeros(self.height, self.width, dtype=torch.long)
            # Simple center mask
            h_start, h_end = self.height // 4, 3 * self.height // 4
            w_start, w_end = self.width // 4, 3 * self.width // 4
            seg[h_start:h_end, w_start:w_end] = 1
            return seg

        seg = Image.open(full_path)
        seg = seg.resize((self.width, self.height), Image.NEAREST)
        seg = torch.from_numpy(np.array(seg)).long()
        return seg

    def _create_inpaint_mask_from_seg(self, seg: torch.Tensor) -> torch.Tensor:
        """Create inpainting mask from segmentation"""
        # Assuming segmentation has garment regions marked
        # Create mask where garment should be inpainted
        inpaint_mask = (seg > 0).float().unsqueeze(0)
        return inpaint_mask

    def _encode_captions(self, product_id: str) -> Tuple[torch.Tensor, str]:
        """Encode product captions"""
        if product_id in self.captions_dict:
            captions = self.captions_dict[product_id]
            # Join all captions but limit total length more aggressively
            caption_text = ". ".join(captions)
        else:
            caption_text = "fashionable garment"

        # More conservative truncation to avoid token length issues
        # CLIP tokenizer has a max length of 77 tokens
        # Being very conservative: ~2-3 chars per token on average
        max_chars = 150  # Very conservative estimate to stay well under 77 tokens
        if len(caption_text) > max_chars:
            # Find a good breaking point (end of sentence or word)
            truncate_point = caption_text.rfind('. ', 0, max_chars)
            if truncate_point == -1:
                truncate_point = caption_text.rfind(' ', 0, max_chars)
            if truncate_point == -1:
                truncate_point = max_chars

            caption_text = caption_text[:truncate_point].rstrip(' .,') + "."

        # Encode
        encoded = self.tokenizer(
            [caption_text],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return encoded, caption_text

    def __getitem__(self, index):
        sample_info = self.samples[index]

        # Load target item data
        target_item = sample_info['target_item']
        target_image = self._load_image(target_item['image_path'])
        target_sketch = self._load_sketch(target_item['image_path'])
        target_seg = self._load_segmentation(target_item['image_path'])
        target_mask = self._create_inpaint_mask_from_seg(target_seg)

        # Encode captions
        target_captions, target_caption_text = self._encode_captions(target_item['product_id'])

        # Load past weeks data
        past_data = []
        temporal_weights = sample_info['temporal_weights']

        for week, weight in zip(sample_info['past_weeks'], temporal_weights):
            # Get items from this past week in same category
            week_items = self.weekly_data[week][sample_info['category']]
            if len(week_items) > 0:
                # Sample one item from past week
                past_item = random.choice(week_items)

                past_image = self._load_image(past_item['image_path'])
                past_sketch = self._load_sketch(past_item['image_path'])
                past_captions, past_caption_text = self._encode_captions(past_item['product_id'])

                past_data.append({
                    'image': past_image,
                    'sketch': past_sketch,
                    'captions': past_captions,
                    'caption_text': past_caption_text,
                    'weight': weight,
                    'week': week
                })

        # Aggregate past conditioning
        if len(past_data) > 0:
            # Weighted average of sketches
            weighted_sketch = torch.zeros_like(target_sketch)
            for data in past_data:
                weighted_sketch += data['weight'] * data['sketch']

            # Combine caption texts with temporal context - with better truncation
            past_captions_list = [data['caption_text'] for data in past_data]

            # Truncate individual captions first to prevent overflow
            max_caption_len = 50  # characters per caption
            truncated_past_captions = [
                cap[:max_caption_len] + "..." if len(cap) > max_caption_len else cap
                for cap in past_captions_list
            ]

            combined_caption = f"trending style: {truncated_past_captions[-1]}"
            if len(truncated_past_captions) > 1:
                # Limit the "evolved from" part
                evolution_part = "; ".join(truncated_past_captions[:-1])
                if len(evolution_part) > 100:  # Limit evolution history
                    evolution_part = evolution_part[:100] + "..."
                combined_caption += f", evolved from: {evolution_part}"

            # Final length check
            if len(combined_caption) > 200:
                combined_caption = combined_caption[:200] + "..."

            # Encode combined caption
            combined_captions = self.tokenizer(
                [combined_caption],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)

            past_conditioning = {
                'weighted_sketch': weighted_sketch,
                'combined_captions': combined_captions,
                'combined_caption_text': combined_caption
            }
        else:
            # No past data, use target as fallback
            past_conditioning = {
                'weighted_sketch': target_sketch,
                'combined_captions': target_captions,
                'combined_caption_text': target_caption_text
            }

        # Since we don't use pose maps, create dummy pose map
        pose_map = torch.zeros(18, self.height, self.width)

        # Get next week's actual text if available
        next_week_actual_text = ""  # Default to empty string instead of None
        if sample_info['next_week_item'] is not None:
            next_week_item = sample_info['next_week_item']
            _, next_week_actual_text = self._encode_captions(next_week_item['product_id'])

        return {
            # Target data
            'image': target_image,
            'im_sketch': target_sketch,
            'im_seg': target_seg,  # Add segmentation mask
            'inpaint_mask': target_mask,
            'pose_map': pose_map,  # Dummy pose map
            'captions': target_captions,
            'original_captions': target_caption_text,

            # Temporal data
            'past_conditioning': past_conditioning,
            'temporal_weights': torch.tensor(temporal_weights, dtype=torch.float32),

            # Next week data
            'next_week_actual_text': next_week_actual_text,

            # Metadata
            'im_name': os.path.basename(target_item['image_path']),
            'product_id': target_item['product_id'],
            'category': target_item['category'],
            'target_week': sample_info['target_week'],
            'past_weeks': sample_info['past_weeks'],
            'next_week_item_id': sample_info['next_week_item']['product_id'] if sample_info['next_week_item'] else "",
            'next_week': sample_info['next_week'] if sample_info['next_week'] else ""
        }