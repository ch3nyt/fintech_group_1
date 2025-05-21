import json
import pathlib
import random
import sys
from typing import Tuple

PROJECT_ROOT = pathlib.Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw
import numpy as np
from torchvision.ops import masks_to_boxes

class SimpleDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 phase: str,
                 tokenizer,
                 sketch_threshold_range: Tuple[int, int] = (20, 127),
                 order: str = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_sketch', 'captions', 'original_captions',
                                         'pose_map', 'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total'),
                 category: Tuple[str] = ('dresses', 'upper_body', 'lower_body'),
                 size: Tuple[int, int] = (512, 384),
                 radius: int = 5,
                 ):

        super(SimpleDataset, self).__init__()
        self.dataroot = pathlib.Path(dataroot_path)
        self.phase = phase
        self.sketch_threshold_range = sketch_threshold_range
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.tokenizer = tokenizer
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        im_names = []
        c_names = []
        dataroot_names = []

        possible_outputs = ['c_name', 'im_name', 'image', 'im_sketch', 'captions', 'original_captions',
                          'pose_map', 'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total']

        assert all(x in possible_outputs for x in outputlist)

        # Load Captions
        with open(self.dataroot / 'captions.json') as f:
            self.captions_dict = json.load(f)
        self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}

        for c in category:
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = self.dataroot / c
            if phase == 'train':
                filename = dataroot / f"{phase}_pairs.txt"
            else:
                filename = dataroot / f"{phase}_pairs_{order}.txt"

            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    if c_name.split('_')[0] not in self.captions_dict:
                        continue

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]

        sketch_threshold = random.randint(self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist or "original_captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            if self.phase == 'train':
                random.shuffle(captions)
            captions = ", ".join(captions)
            original_captions = captions

        if "captions" in self.outputlist:
            cond_input = self.tokenizer([captions], max_length=self.tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids
            cond_input = cond_input.squeeze(0)
            max_length = cond_input.shape[-1]
            uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            ).input_ids.squeeze(0)
            captions = cond_input

        if "image" in self.outputlist:
            image = Image.open(dataroot / 'images' / im_name)
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "im_sketch" in self.outputlist:
            if "unpaired" == self.order and self.phase == 'test':
                im_sketch = Image.open(
                    dataroot / 'im_sketch' / 'unpaired' / f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}')
            else:
                im_sketch = Image.open(dataroot / 'im_sketch' / 'paired' / c_name.replace(".jpg", ".png"))

            im_sketch = im_sketch.resize((self.width, self.height))
            im_sketch = ImageOps.invert(im_sketch)
            im_sketch = im_sketch.point(lambda p: 255 if p > sketch_threshold else 0)
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [-1,1]
            im_sketch = 1 - im_sketch

        # Generate dummy pose map and parse array for compatibility
        if "pose_map" in self.outputlist:
            # Create a dummy pose map with zeros
            pose_map = torch.zeros(18, self.height, self.width)  # 18 keypoints as in COCO format

        if "parse_array" in self.outputlist:
            # Create a dummy parse array (segmentation map)
            parse_array = np.zeros((self.height, self.width), dtype=np.uint8)
            # Set some basic regions (you might want to adjust these based on your needs)
            parse_array[self.height//4:self.height*3//4, self.width//4:self.width*3//4] = 1  # body region

        if "im_mask" in self.outputlist or "inpaint_mask" in self.outputlist or "parse_mask_total" in self.outputlist:
            # Create masks based on the parse array
            parse_mask = (parse_array > 0).astype(np.float32)
            parse_mask_total = parse_mask.copy()

            # Create inpaint mask (1 for regions to be inpainted)
            inpaint_mask = 1 - parse_mask
            inpaint_mask = torch.from_numpy(inpaint_mask).unsqueeze(0)

            # Create image mask
            im_mask = image * (1 - inpaint_mask.repeat(3, 1, 1))

        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self):
        return len(self.c_names)