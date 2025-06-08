import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

def parse_caption_file(file_path):
    """Parses the caption file to extract image metadata."""
    with open(file_path, 'r') as f:
        content = f.read()

    image_blocks = content.strip().split('Image: ')
    extracted_data = []

    for block in image_blocks:
        if not block.strip():
            continue

        data = {}
        data['image_name'] = block.split('\n')[0].strip()

        try:
            data['product_id'] = re.search(r"Product ID: (.*?)\n", block).group(1).strip()
            data['category'] = re.search(r"Category: (.*?)\n", block).group(1).strip()
            data['ref_product_id'] = re.search(r"Actual Text Source Product: (.*?)\n", block).group(1).strip()
            data['ref_week'] = re.search(r"Actual Text Source Week: (.*?)\n", block).group(1).strip()
            extracted_data.append(data)
        except AttributeError:
            print(f"Skipping block due to missing data: {data.get('image_name', 'N/A')}")
            continue

    return extracted_data

def reorganize_images(captions_file, source_dir, dest_dir):
    """Reorganizes images based on the parsed caption data."""
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # 1. Create destination directory
    dest_path.mkdir(exist_ok=True)
    print(f"Destination directory created at: {dest_path.resolve()}")

    # 2. Parse the caption file
    image_metadata = parse_caption_file(captions_file)
    if not image_metadata:
        print("No metadata extracted. Exiting.")
        return

    print(f"Found {len(image_metadata)} entries to process.")

    processed_count = 0
    not_found_count = 0

    # 3. Create category subdirectories and process images
    categories = {item['category'] for item in image_metadata}
    for category in categories:
        (dest_path / category).mkdir(exist_ok=True)

    print(f"Created category subdirectories in {dest_path}")

    # 4. Iterate through metadata and copy files
    for item in tqdm(image_metadata, desc="Reorganizing images"):
        original_id = item['product_id']
        ref_id = item['ref_product_id']
        category = item['category']
        week = item['ref_week']

        # Convert week format from "2020-32" to "2020-week32"
        if '-' in week and not week.startswith('week'):
            year, week_num = week.split('-')
            formatted_week = f"{year}-week{week_num}"
        else:
            formatted_week = week

        # Construct the source path for the reference image
        source_file_path = source_path / category / formatted_week / f"{ref_id}.jpg"

        # Construct the new filename and destination path
        new_filename = f"{original_id}_{ref_id}.jpg"
        dest_file_path = dest_path / category / new_filename

        if source_file_path.exists():
            shutil.copy(source_file_path, dest_file_path)
            processed_count += 1
        else:
            print(f"Source file not found: {source_file_path}")
            not_found_count += 1

    # 5. Final summary
    print("\n--- Reorganization Complete ---")
    print(f"Successfully processed and copied: {processed_count} images")
    print(f"Files not found (skipped): {not_found_count}")

if __name__ == "__main__":
    # Corrected base path to be inside the project directory
    project_base_dir = Path('/root/multimodal-garment-designer')

    captions_file = project_base_dir / 'inference_result_example/captions_used.txt'
    source_image_dir = project_base_dir / 'dataset_vitonhd_format/test/images'
    destination_dir = project_base_dir / 'ref_image'

    reorganize_images(captions_file, source_image_dir, destination_dir)