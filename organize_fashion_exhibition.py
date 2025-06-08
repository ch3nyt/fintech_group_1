#!/usr/bin/env python3
import os
import shutil
import re
import glob

# Define the image lists as provided by the user
image_data = {
    'top5acc': [
        '0552716001_pred_0150',
        '0639448011_pred_0152',
        '0833098001_pred_0241',
        '0843380004_pred_0121',
        '0852541001_pred_0001',
        '0904734001_pred_0060'
    ],
    'top5gfb': [
        '0797078018_pred_0069',
        '0852369005_pred_0006',
        '0862970001_pred_0125',
        '0878013001_pred_0035',
        '0909185001_pred_0037',
        '0909371001_pred_0189'
    ],
    'top5glb': [
        '0706016003_pred_0043',
        '0714790020_pred_0222',
        '0751471001_pred_0134',
        '0865799006_pred_0224',
        '0918292001_pred_0160',
        '0926502001_pred_0192',
        '0929275001_pred_0254'
    ],
    'top5gub': [
        '0827968001_pred_0015',
        '0896152002_pred_0075',
        '0903049003_pred_0016',
        '0915529003_pred_0138',
        '0896169002_pred_0108',
        '0918522001_pred_0225',
        '0933838001_pred_0017'
    ],
    'top5shoe': [
        '0754357005_pred_0080',
        '0808684002_pred_0023',
        '0889713001_pred_0051',
        '0893798001_pred_0140',
        '0903487001_pred_0260',
        '0916256001_pred_0021'
    ]
}

def extract_full_product_info(captions_file, image_name):
    """Extract complete product information from the captions file for a specific image"""
    with open(captions_file, 'r') as f:
        content = f.read()

    # Look for the complete image entry block
    pattern = f"Image: {image_name}\\.jpg.*?(?=Image: |$)"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        return match.group(0).strip()
    return f"Product information not found for {image_name}"

def find_and_copy_additional_files(product_id, category, product_dir, base_path):
    """Find and copy im_sketch, im_seg, and original image files for the product"""
    # Paths to search for additional files
    im_sketch_base = os.path.join(base_path, 'dataset_vitonhd_format', 'test', 'im_sketch', category)
    im_seg_base = os.path.join(base_path, 'dataset_vitonhd_format', 'test', 'im_seg', category)
    images_base = os.path.join(base_path, 'dataset_vitonhd_format', 'test', 'images', category)

    # Find im_sketch file (searches through all week folders)
    sketch_found = False
    if os.path.exists(im_sketch_base):
        for week_folder in os.listdir(im_sketch_base):
            week_path = os.path.join(im_sketch_base, week_folder)
            if os.path.isdir(week_path):
                # Look for the product file in this week folder
                sketch_file = os.path.join(week_path, f"{product_id}.png")
                if os.path.exists(sketch_file):
                    target_sketch = os.path.join(product_dir, f"{product_id}_im_sketch.png")
                    shutil.copy2(sketch_file, target_sketch)
                    print(f"    Copied sketch: {product_id}.png -> {product_id}_im_sketch.png")
                    sketch_found = True
                    break

    if not sketch_found:
        print(f"    Warning: No im_sketch file found for product {product_id}")

    # Find im_seg file (searches through all week folders)
    seg_found = False
    if os.path.exists(im_seg_base):
        for week_folder in os.listdir(im_seg_base):
            week_path = os.path.join(im_seg_base, week_folder)
            if os.path.isdir(week_path):
                # Look for the product file in this week folder (try both naming patterns)
                seg_patterns = [
                    f"{product_id}_final_seg.png",
                    f"{product_id}.png"
                ]

                for pattern in seg_patterns:
                    seg_file = os.path.join(week_path, pattern)
                    if os.path.exists(seg_file):
                        target_seg = os.path.join(product_dir, f"{product_id}_im_seg.png")
                        shutil.copy2(seg_file, target_seg)
                        print(f"    Copied seg: {pattern} -> {product_id}_im_seg.png")
                        seg_found = True
                        break

                if seg_found:
                    break

    if not seg_found:
        print(f"    Warning: No im_seg file found for product {product_id}")

    # Find original image file (searches through all week folders)
    original_found = False
    if os.path.exists(images_base):
        for week_folder in os.listdir(images_base):
            week_path = os.path.join(images_base, week_folder)
            if os.path.isdir(week_path):
                # Look for the product file in this week folder (try different extensions)
                image_patterns = [
                    f"{product_id}.jpg",
                    f"{product_id}.png",
                    f"{product_id}.jpeg"
                ]

                for pattern in image_patterns:
                    image_file = os.path.join(week_path, pattern)
                    if os.path.exists(image_file):
                        # Get the original extension and use .png for consistency
                        target_original = os.path.join(product_dir, f"{product_id}_original.png")
                        shutil.copy2(image_file, target_original)
                        print(f"    Copied original: {pattern} -> {product_id}_original.png")
                        original_found = True
                        break

                if original_found:
                    break

    if not original_found:
        print(f"    Warning: No original image file found for product {product_id}")

def main():
    # Paths
    base_path = '/root/multimodal-garment-designer'
    source_predictions = f'{base_path}/temporal_vitonhd_dpo_inference/temporal_vitonhd_dpo_inference_20250603_165701_For_Kuo/predictions'
    captions_file = f'{base_path}/temporal_vitonhd_dpo_inference/temporal_vitonhd_dpo_inference_20250603_165701_For_Kuo/captions_used.txt'
    target_base = f'{base_path}/Fashion_Distill_Exhibition'

    # Create the base exhibition directory
    os.makedirs(target_base, exist_ok=True)

    # Process each category
    for category, images in image_data.items():
        if not images:  # Skip empty categories
            continue

        print(f"Processing category: {category}")

        # Create category directory
        category_dir = os.path.join(target_base, category)
        os.makedirs(category_dir, exist_ok=True)

        for image_name in images:
            # Extract product ID from image name (everything before '_pred_')
            product_id = image_name.split('_pred_')[0]

            # Create product directory within category
            product_dir = os.path.join(category_dir, product_id)
            os.makedirs(product_dir, exist_ok=True)

            print(f"  Processing product: {product_id}")

            # Source image path
            source_image = os.path.join(source_predictions, category, f"{image_name}.jpg")
            target_image = os.path.join(product_dir, f"{image_name}.jpg")

            # Copy image if it exists
            if os.path.exists(source_image):
                shutil.copy2(source_image, target_image)
                print(f"    Copied: {image_name}.jpg")
            else:
                print(f"    Warning: Image not found: {source_image}")

            # Extract and save complete product information
            product_info = extract_full_product_info(captions_file, image_name)
            caption_file = os.path.join(product_dir, f"{product_id}.txt")

            # If caption file already exists, append to it; otherwise create new
            mode = 'a' if os.path.exists(caption_file) else 'w'
            with open(caption_file, mode) as f:
                if mode == 'a':
                    f.write(f"\n\n{'='*60}\n")
                f.write(product_info)

            print(f"    Saved product info for: {product_id}")

            # Find and copy additional files (im_sketch, im_seg, and original image)
            find_and_copy_additional_files(product_id, category, product_dir, base_path)

    print("\nOrganization complete!")
    print(f"Files organized in: {target_base}")

    # Show the final structure
    print("\nFinal directory structure:")
    for root, dirs, files in os.walk(target_base):
        level = root.replace(target_base, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            print(f'{subindent}{file}')

if __name__ == "__main__":
    main()