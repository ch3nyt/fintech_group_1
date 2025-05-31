#!/usr/bin/env python3
import os
import shutil
import json
from pathlib import Path
import re

def get_week_info(week_folder_name):
    """Extract year and week number from folder name like '2020-week39'"""
    match = re.match(r'(\d{4})-week(\d+)', week_folder_name)
    if match:
        year = int(match.group(1))
        week = int(match.group(2))
        return year, week
    return None, None

def get_split_for_week(week_folder_name):
    """Determine which split (train/val/test) a week belongs to"""
    year, week = get_week_info(week_folder_name)
    if year is None or week is None:
        return None

    # Convert to a comparable format: YYYYWW
    week_id = year * 100 + week

    # Define ranges
    train_start = 2018 * 100 + 38  # 2018-week38
    train_end = 2020 * 100 + 7     # 2020-week7
    val_start = 2020 * 100 + 8     # 2020-week8
    val_end = 2020 * 100 + 22      # 2020-week22
    test_start = 2020 * 100 + 23   # 2020-week23
    test_end = 2020 * 100 + 39     # 2020-week39

    if train_start <= week_id <= train_end:
        return "train"
    elif val_start <= week_id <= val_end:
        return "val"
    elif test_start <= week_id <= test_end:
        return "test"
    else:
        return None  # Skip weeks outside our range

def create_target_structure():
    """Create the target directory structure with train/val/test splits"""
    base_dir = Path("/root/multimodal-garment-designer/dataset_vitonhd_format")

    # Create directories for each split
    splits = ["train", "val", "test"]
    subdirs = ["images", "im_sketch", "im_seg"]

    for split in splits:
        for subdir in subdirs:
            (base_dir / split / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir

def copy_images_with_structure(source_dir, target_base_dir):
    """Copy image files maintaining the original folder structure within splits"""
    source_path = Path(source_dir)
    target_base = Path(target_base_dir)

    split_counts = {"train": 0, "val": 0, "test": 0}

    # Iterate through category folders (top5gub, top5glb, etc.)
    for category_folder in source_path.iterdir():
        if category_folder.is_dir() and category_folder.name.startswith('top5'):
            print(f"Processing category: {category_folder.name}")

            # Iterate through week folders
            for week_folder in category_folder.iterdir():
                if week_folder.is_dir():
                    split = get_split_for_week(week_folder.name)
                    if split is None:
                        continue  # Skip weeks outside our range

                    # Create target directory structure
                    target_category_dir = target_base / split / "images" / category_folder.name
                    target_week_dir = target_category_dir / week_folder.name
                    target_week_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all image files from this week folder
                    for file_path in week_folder.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            target_file = target_week_dir / file_path.name

                            if not target_file.exists():
                                shutil.copy2(file_path, target_file)
                                split_counts[split] += 1

                                if split_counts[split] % 100 == 0:
                                    print(f"  {split}: Copied {split_counts[split]} files...")

    for split, count in split_counts.items():
        print(f"Total {split} images copied: {count}")

    return split_counts

def copy_other_data(source_dir, target_base_dir, data_type):
    """Copy sketch or segmentation files maintaining the original folder structure"""
    source_path = Path(source_dir)
    target_base = Path(target_base_dir)

    split_counts = {"train": 0, "val": 0, "test": 0}

    # Iterate through category folders
    for category_folder in source_path.iterdir():
        if category_folder.is_dir() and category_folder.name.startswith('top5'):
            print(f"Processing {data_type} for category: {category_folder.name}")

            # Iterate through week folders
            for week_folder in category_folder.iterdir():
                if week_folder.is_dir():
                    split = get_split_for_week(week_folder.name)
                    if split is None:
                        continue  # Skip weeks outside our range

                    # Create target directory structure
                    target_category_dir = target_base / split / data_type / category_folder.name
                    target_week_dir = target_category_dir / week_folder.name
                    target_week_dir.mkdir(parents=True, exist_ok=True)

                    # Copy all files from this week folder
                    for file_path in week_folder.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            target_file = target_week_dir / file_path.name

                            if not target_file.exists():
                                shutil.copy2(file_path, target_file)
                                split_counts[split] += 1

                                if split_counts[split] % 100 == 0:
                                    print(f"  {split}: Copied {split_counts[split]} {data_type} files...")

    for split, count in split_counts.items():
        print(f"Total {split} {data_type} files copied: {count}")

    return split_counts

def collect_captions_from_output(captions_source_dir, output_file):
    """Collect caption data from the actual JSON files in the output directory"""
    source_path = Path(captions_source_dir)
    captions_data = {}

    print("Processing captions from JSON files...")

    # Iterate through category folders
    for category_folder in source_path.iterdir():
        if category_folder.is_dir() and category_folder.name.startswith('top5'):
            print(f"Processing captions from category: {category_folder.name}")

            # Iterate through week folders
            for week_folder in category_folder.iterdir():
                if week_folder.is_dir():
                    # Look for JSON files containing descriptions
                    for file_path in week_folder.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == '.json' and 'descriptions' in file_path.name:
                            try:
                                # Read the JSON file
                                with open(file_path, 'r') as f:
                                    week_captions = json.load(f)

                                # Add captions to our main dictionary, skipping duplicates
                                for product_name, captions in week_captions.items():
                                    if product_name not in captions_data:
                                        captions_data[product_name] = captions
                                    # else: skip duplicate products as requested

                            except (json.JSONDecodeError, Exception) as e:
                                print(f"Error reading {file_path}: {e}")
                                continue

    # Write captions to JSON file
    with open(output_file, 'w') as f:
        json.dump(captions_data, f, indent=2)

    print(f"Created captions file with {len(captions_data)} unique products")
    return len(captions_data)

def collect_captions_from_images(target_base_dir, output_file):
    """Collect caption data based on product names from image files"""
    target_base = Path(target_base_dir)
    captions_data = {}

    # Process all splits to get unique product names
    splits = ["train", "val", "test"]

    for split in splits:
        images_dir = target_base / split / "images"
        if not images_dir.exists():
            continue

        print(f"Processing captions from {split} split...")

        # Iterate through category folders
        for category_folder in images_dir.iterdir():
            if category_folder.is_dir() and category_folder.name.startswith('top5'):
                category_type = category_folder.name.replace('top5', '')

                # Iterate through week folders
                for week_folder in category_folder.iterdir():
                    if week_folder.is_dir():
                        # Process all image files
                        for file_path in week_folder.iterdir():
                            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                # Extract product name (filename without extension)
                                product_name = file_path.stem

                                # Skip if we already have this product
                                if product_name in captions_data:
                                    continue

                                # Create captions based on category
                                if category_type == 'gub':
                                    category_desc = "upper body garment"
                                elif category_type == 'glb':
                                    category_desc = "lower body garment"
                                elif category_type == 'gfb':
                                    category_desc = "full body garment"
                                elif category_type == 'acc':
                                    category_desc = "accessory"
                                elif category_type == 'shoe':
                                    category_desc = "footwear"
                                elif category_type == 'underwear':
                                    category_desc = "underwear"
                                else:
                                    category_desc = "clothing item"

                                captions_data[product_name] = [
                                    f"{category_desc.title()}.",
                                    f"Fashion {category_desc}.",
                                    f"Stylish {category_desc} for everyday wear."
                                ]

    # Write captions to JSON file
    with open(output_file, 'w') as f:
        json.dump(captions_data, f, indent=2)

    print(f"Created captions file with {len(captions_data)} unique products")
    return len(captions_data)

def main():
    print("Starting dataset reorganization to vitonhd format with train/val/test splits...")

    # Source directories
    images_source = "/root/multimodal-garment-designer/top5"
    sketches_source = "/root/multimodal-garment-designer/output_pidinet_sketch"
    segmentation_source = "/root/multimodal-garment-designer/huggingface-cloth-segmentation/output"
    captions_source = "/root/multimodal-garment-designer/output"

    # Create target structure
    base_dir = create_target_structure()

    print(f"Created target directory structure at: {base_dir}")

    # Copy images with folder structure
    print("\n1. Copying images...")
    image_counts = copy_images_with_structure(images_source, base_dir)

    # Copy sketches with folder structure
    print("\n2. Copying sketches...")
    sketch_counts = copy_other_data(sketches_source, base_dir, "im_sketch")

    # Copy segmentation files with folder structure
    print("\n3. Copying segmentation files...")
    seg_counts = copy_other_data(segmentation_source, base_dir, "im_seg")

    # Create captions file
    print("\n4. Creating captions file...")
    caption_count = collect_captions_from_output(captions_source, base_dir / "captions.json")

    print(f"\nReorganization complete!")
    print(f"Split summary:")
    for split in ["train", "val", "test"]:
        print(f"- {split.upper()}:")
        print(f"  Images: {image_counts.get(split, 0)}")
        print(f"  Sketches: {sketch_counts.get(split, 0)}")
        print(f"  Segmentation: {seg_counts.get(split, 0)}")
    print(f"- Unique products in captions: {caption_count}")
    print(f"\nData organized in: {base_dir}")

if __name__ == "__main__":
    main()