#!/usr/bin/env python3
### DO NOT REMOVE THIS COMMENT AT THE TOP.

# You are in /data/superstimuli_group
# There is a folder called beautiful_images
# Within beautiful_images, make one called "square" where it's literally just every image cropped to be square (but otherwise the same resolution as the smaller of the width or the height, as a square)
# Afterward, make one called "2056" where it's literally just every image but 2056x2056
# Afterward, make one called "1024" where it's literally just every image but 1024x1024
# Afterward, make one called "256" where it's literally just every image but 256x256



"""
Process images from beautiful_images folder:
1. Create 'square' folder with center-cropped square images
2. Create '2056' folder with 2056x2056 resized images
3. Create '1024' folder with 1024x1024 resized images
4. Create '256' folder with 256x256 resized images
5. Generate JSON files listing all images in wikiart_final.json format
"""

import os
import json
from PIL import Image
from pathlib import Path

def center_crop_square(img):
    """Crop image to square using the smaller dimension."""
    width, height = img.size
    size = min(width, height)

    # Calculate crop box for center crop
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    return img.crop((left, top, right, bottom))

def process_images():
    """Process all images in beautiful_images folder."""
    source_dir = Path("/data/superstimuli_group/beautiful_images")

    # Define output directories
    output_dirs = {
        'square': source_dir / 'square',
        '2056': source_dir / '2056',
        '1024': source_dir / '1024',
        '256': source_dir / '256'
    }

    # Create output directories
    for dir_path in output_dirs.values():
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in source_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"\nFound {len(image_files)} images to process\n")

    # Process each image
    for idx, img_path in enumerate(image_files, 1):
        print(f"Processing {idx}/{len(image_files)}: {img_path.name}")

        try:
            with Image.open(img_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 1. Create square (center-cropped)
                square_img = center_crop_square(img)
                output_path = output_dirs['square'] / img_path.name
                # Use JPEG format, change extension if needed
                if output_path.suffix.lower() == '.png':
                    output_path = output_path.with_suffix('.jpg')
                square_img.save(output_path, 'JPEG', quality=95)
                print(f"  - Saved square: {square_img.size}")

                # 2. Resize to 2056x2056
                img_2056 = square_img.resize((2056, 2056), Image.Resampling.LANCZOS)
                output_path = output_dirs['2056'] / img_path.name
                if output_path.suffix.lower() == '.png':
                    output_path = output_path.with_suffix('.jpg')
                img_2056.save(output_path, 'JPEG', quality=95)
                print(f"  - Saved 2056x2056")

                # 3. Resize to 1024x1024
                img_1024 = square_img.resize((1024, 1024), Image.Resampling.LANCZOS)
                output_path = output_dirs['1024'] / img_path.name
                if output_path.suffix.lower() == '.png':
                    output_path = output_path.with_suffix('.jpg')
                img_1024.save(output_path, 'JPEG', quality=95)
                print(f"  - Saved 1024x1024")

                # 4. Resize to 256x256
                img_256 = square_img.resize((256, 256), Image.Resampling.LANCZOS)
                output_path = output_dirs['256'] / img_path.name
                if output_path.suffix.lower() == '.png':
                    output_path = output_path.with_suffix('.jpg')
                img_256.save(output_path, 'JPEG', quality=95)
                print(f"  - Saved 256x256")

        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            continue

    print("\n=== Processing Complete ===")
    for name, dir_path in output_dirs.items():
        count = len(list(dir_path.glob('*')))
        print(f"{name}: {count} images")

    # Generate one big mixed JSON file
    print("\n=== Generating Mixed JSON File ===")
    generate_mixed_json_file(output_dirs)

def generate_mixed_json_file(output_dirs):
    """Generate one big JSON file with all images from all resolutions mixed together."""
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = []

    # Collect all images from all directories
    for name, dir_path in output_dirs.items():
        image_files = sorted([f for f in dir_path.iterdir()
                             if f.is_file() and f.suffix.lower() in image_extensions])
        all_images.extend(image_files)
        print(f"Collected {len(image_files)} images from {name}/")

    # Create the JSON structure
    json_data = {
        "Images": [
            {"images": [str(img_path.absolute())]}
            for img_path in all_images
        ]
    }

    # Save to JSON file in /richard/small_runs/
    output_path = Path("/data/superstimuli_group/richard/small_runs/resolution_test_final.json")
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\nGenerated {output_path}")
    print(f"  - Total images: {len(all_images)}")
    print(f"  - Images per resolution: {len(all_images) // len(output_dirs)}")

if __name__ == "__main__":
    process_images()
