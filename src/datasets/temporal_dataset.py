import json
import os
import pathlib
import random
import sys
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes
import cv2

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.posemap import get_coco_body25_mapping, kpoint_to_heatmap


class TemporalGarmentDataset(data.Dataset):
    """
    Dataset for temporal garment prediction with weighted past weeks
    """
    def __init__(
        self,
        dataroot_path: str,
        phase: str,
        tokenizer,
        num_past_weeks: int = 4,
        temporal_weight_decay: float = 0.8,  # Weight decay for older weeks
        sketch_threshold_range: Tuple[int, int] = (20, 127),
        size: Tuple[int, int] = (512, 384),
        radius: int = 5,
    ):
        super(TemporalGarmentDataset, self).__init__()

        self.dataroot = pathlib.Path(dataroot_path)
        self.phase = phase
        self.num_past_weeks = num_past_weeks
        self.temporal_weight_decay = temporal_weight_decay
        self.sketch_threshold_range = sketch_threshold_range
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load weekly data structure
        self.weekly_data = self._load_weekly_data()
        self.samples = self._create_temporal_samples()

    def _load_weekly_data(self) -> Dict[str, List[Dict]]:
        """
        Load weekly garment data
        Expected structure:
        - weekly_data.json: {"week_1": [{"image": "path", "style_text": "description", "garment_type": "dress", ...}]}
        """
        weekly_file = self.dataroot / f"{self.phase}_weekly_data.json"
        with open(weekly_file, 'r') as f:
            weekly_data = json.load(f)

        # Sort weeks chronologically
        sorted_weeks = sorted(weekly_data.keys(), key=lambda x: int(x.split('_')[1]))
        return {week: weekly_data[week] for week in sorted_weeks}

    def _create_temporal_samples(self) -> List[Tuple]:
        """
        Create training samples with temporal context
        Each sample contains: (target_week, past_weeks_data, temporal_weights)
        """
        samples = []
        weeks = list(self.weekly_data.keys())

        for i in range(self.num_past_weeks, len(weeks)):
            target_week = weeks[i]
            past_weeks = weeks[max(0, i-self.num_past_weeks):i]

            # Calculate temporal weights (closer weeks get higher weights)
            weights = [self.temporal_weight_decay ** (len(past_weeks) - j - 1)
                      for j in range(len(past_weeks))]
            weights = np.array(weights) / sum(weights)  # Normalize

            for target_item in self.weekly_data[target_week]:
                samples.append((target_week, past_weeks, weights, target_item))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        target_week, past_weeks, temporal_weights, target_item = self.samples[index]

        # Load target garment data
        target_data = self._load_garment_data(target_item, target_week)

        # Load and weight past weeks data
        past_data = []
        for week, weight in zip(past_weeks, temporal_weights):
            week_items = self.weekly_data[week]
            # Sample representative item from past week (could be random or style-matched)
            past_item = random.choice(week_items)
            past_garment_data = self._load_garment_data(past_item, week)
            past_garment_data['temporal_weight'] = weight
            past_data.append(past_garment_data)

        # Create aggregated conditioning from past weeks
        aggregated_conditioning = self._aggregate_past_conditioning(past_data, temporal_weights)

        # Combine with target
        result = {
            **target_data,
            'past_conditioning': aggregated_conditioning,
            'temporal_weights': torch.tensor(temporal_weights, dtype=torch.float32),
            'target_week': target_week,
            'past_weeks': past_weeks,
        }

        return result

    def _load_garment_data(self, item_data: Dict, week: str) -> Dict:
        """Load individual garment data (image, pose, sketch, etc.)"""
        week_dir = self.dataroot / week

        # Load image
        image_path = week_dir / 'images' / item_data['image']
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.width, self.height))
        image = self.transform(image)

        # Load or generate pose map
        pose_path = week_dir / 'poses' / item_data['image'].replace('.jpg', '_pose.json')
        if pose_path.exists():
            pose_map = self._load_pose_map(pose_path)
        else:
            # Generate dummy pose map
            pose_map = torch.zeros(18, self.height, self.width)

        # Load or generate sketch
        sketch_path = week_dir / 'sketches' / item_data['image'].replace('.jpg', '.png')
        if sketch_path.exists():
            sketch = self._load_sketch(sketch_path)
        else:
            # Generate sketch from image (simple edge detection)
            sketch = self._generate_sketch_from_image(image)

        # Load or generate masks
        mask_path = week_dir / 'masks' / item_data['image'].replace('.jpg', '.png')
        if mask_path.exists():
            inpaint_mask = self._load_mask(mask_path)
        else:
            # Generate simple center mask
            inpaint_mask = self._generate_simple_mask()

        # Encode style text
        style_text = item_data.get('style_text', 'casual garment')
        captions = self.tokenizer([style_text],
                                max_length=self.tokenizer.model_max_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt").input_ids.squeeze(0)

        return {
            'image': image,
            'pose_map': pose_map,
            'im_sketch': sketch,
            'inpaint_mask': inpaint_mask,
            'captions': captions,
            'original_captions': style_text,
            'im_name': item_data['image'],
            'garment_type': item_data.get('garment_type', 'unknown')
        }

    def _load_pose_map(self, pose_path: pathlib.Path) -> torch.Tensor:
        """Load pose keypoints and convert to heatmap"""
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        keypoints = np.array(pose_data['keypoints']).reshape(-1, 3)
        pose_map = torch.zeros(18, self.height, self.width)

        for i, (x, y, v) in enumerate(keypoints[:18]):
            if v > 0:  # visible keypoint
                pose_map[i] = kpoint_to_heatmap(
                    np.array([x, y]), (self.height, self.width), self.radius
                )

        return pose_map

    def _load_sketch(self, sketch_path: pathlib.Path) -> torch.Tensor:
        """Load and process sketch"""
        sketch_threshold = random.randint(*self.sketch_threshold_range)

        sketch = Image.open(sketch_path).convert('L')
        sketch = sketch.resize((self.width, self.height))
        sketch = ImageOps.invert(sketch)
        sketch = sketch.point(lambda p: 255 if p > sketch_threshold else 0)
        sketch = transforms.functional.to_tensor(sketch)
        sketch = 1 - sketch

        return sketch

    def _load_mask(self, mask_path: pathlib.Path) -> torch.Tensor:
        """Load inpainting mask"""
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.width, self.height))
        mask = transforms.functional.to_tensor(mask)

        return mask

    def _generate_sketch_from_image(self, image: torch.Tensor) -> torch.Tensor:
        """Generate sketch using edge detection"""
        # Convert to numpy
        img_np = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Edge detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Convert back to tensor
        sketch = torch.from_numpy(edges).float() / 255.0
        sketch = sketch.unsqueeze(0)

        return sketch

    def _generate_simple_mask(self) -> torch.Tensor:
        """Generate simple center mask for inpainting"""
        mask = torch.zeros(1, self.height, self.width)
        h_start, h_end = self.height // 4, 3 * self.height // 4
        w_start, w_end = self.width // 4, 3 * self.width // 4
        mask[0, h_start:h_end, w_start:w_end] = 1.0

        return mask

    def _aggregate_past_conditioning(self, past_data: List[Dict], weights: np.ndarray) -> Dict:
        """Aggregate conditioning information from past weeks using temporal weights"""

        # Weighted average of pose maps
        weighted_pose = torch.zeros_like(past_data[0]['pose_map'])
        for data, weight in zip(past_data, weights):
            weighted_pose += weight * data['pose_map']

        # Weighted average of sketches
        weighted_sketch = torch.zeros_like(past_data[0]['im_sketch'])
        for data, weight in zip(past_data, weights):
            weighted_sketch += weight * data['im_sketch']

        # Combine style texts with weights (prioritize recent weeks)
        style_texts = [data['original_captions'] for data in past_data]
        # Use most recent style as primary, others as context
        combined_style = f"trending style: {style_texts[-1]}"
        if len(style_texts) > 1:
            combined_style += f", influenced by: {', '.join(style_texts[:-1])}"

        # Encode combined style text
        combined_captions = self.tokenizer([combined_style],
                                         max_length=self.tokenizer.model_max_length,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt").input_ids.squeeze(0)

        return {
            'weighted_pose': weighted_pose,
            'weighted_sketch': weighted_sketch,
            'combined_captions': combined_captions,
            'combined_style_text': combined_style
        }