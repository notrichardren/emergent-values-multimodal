#!/usr/bin/env python3
"""
Script to randomly sample images from multiple datasets and create a unified JSONL file.
Each entry contains: path (full path), class (if applicable), and class_name (if applicable).
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
import shutil
import io

# Set random seed for reproducibility
random.seed(42)

OUTPUT_JSONL = "/data/superstimuli_group/richard/big_run_2/sampled_images.jsonl"
WIKIART_OUTPUT_DIR = "/data/superstimuli_group/richard/big_run_2/wikiart"
FOOD101_OUTPUT_DIR = "/data/superstimuli_group/richard/big_run_2/food101"

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')):
    """Recursively get all image files in a directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).rglob(f'*{ext}'))
    return [str(f) for f in image_files]

def sample_from_class_folders(base_path, num_classes, images_per_class, total_images):
    """Sample images evenly from class-based folder structure."""
    results = []

    # Get all class directories
    class_dirs = [d for d in Path(base_path).iterdir() if d.is_dir()]
    print(f"Found {len(class_dirs)} classes in {base_path}")

    # Randomly select classes
    selected_classes = random.sample(class_dirs, min(num_classes, len(class_dirs)))

    images_per_class_target = total_images // len(selected_classes)

    for class_dir in selected_classes:
        class_name = class_dir.name

        # Get all images in this class
        class_images = get_image_files(class_dir)

        if not class_images:
            print(f"Warning: No images found in {class_dir}")
            continue

        # Sample images from this class
        num_to_sample = min(images_per_class_target, len(class_images))
        sampled_images = random.sample(class_images, num_to_sample)

        for img_path in sampled_images:
            results.append({
                "path": img_path,
                "class": class_name,
                "class_name": class_name.replace('_', ' ')
            })

    # If we didn't get enough images, sample more from remaining classes
    if len(results) < total_images:
        remaining = total_images - len(results)
        all_images = []
        for class_dir in selected_classes:
            class_name = class_dir.name
            class_images = get_image_files(class_dir)
            for img_path in class_images:
                if img_path not in [r['path'] for r in results]:
                    all_images.append({
                        "path": img_path,
                        "class": class_name,
                        "class_name": class_name.replace('_', ' ')
                    })

        if all_images:
            additional = random.sample(all_images, min(remaining, len(all_images)))
            results.extend(additional)

    return results[:total_images]

def sample_imagenet_a():
    """Sample 500 images from 100 random classes in ImageNet-A."""
    print("\n=== Sampling ImageNet-A ===")
    base_path = "/data/superstimuli_group/imagenet-a/imagenet-a"
    return sample_from_class_folders(base_path, num_classes=100, images_per_class=5, total_images=500)

def sample_imagenet_o():
    """Sample 500 images from 100 random classes in ImageNet-O."""
    print("\n=== Sampling ImageNet-O ===")
    base_path = "/data/superstimuli_group/imagenet-o/imagenet-o"
    return sample_from_class_folders(base_path, num_classes=100, images_per_class=5, total_images=500)

def sample_imagenet_val():
    """Sample 3000 images from 300 random classes in ImageNet Val."""
    print("\n=== Sampling ImageNet Val ===")
    base_path = "/data/superstimuli_group/imagenet_val"
    return sample_from_class_folders(base_path, num_classes=300, images_per_class=10, total_images=3000)

def sample_species():
    """Sample ~400 images from 60 random classes in Species dataset."""
    print("\n=== Sampling Species ===")
    base_path = "/data/superstimuli_group/species_data/species_for_finegrained/species_splits/test/iid"
    return sample_from_class_folders(base_path, num_classes=60, images_per_class=7, total_images=400)

def sample_wikiart():
    """Sample 600 images from WikiArt parquet files, evenly across genres."""
    print("\n=== Sampling WikiArt ===")
    parquet_dir = "/data/superstimuli_group/wikiart"
    ensure_dir(WIKIART_OUTPUT_DIR)

    # Read all parquet files
    parquet_files = list(Path(parquet_dir).glob("train-*.parquet"))
    print(f"Found {len(parquet_files)} WikiArt parquet files")

    all_data = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total WikiArt images: {len(df)}")

    # Check if there's a genre or style column
    print(f"WikiArt columns: {df.columns.tolist()}")

    # Assuming there's a 'style' or 'genre' column - adjust based on actual schema
    genre_col = None
    for col in ['genre', 'style', 'artist', 'label']:
        if col in df.columns:
            genre_col = col
            break

    results = []

    if genre_col:
        # Sample evenly across genres
        genres = df[genre_col].unique()
        print(f"Found {len(genres)} unique {genre_col}s")

        images_per_genre = 600 // len(genres)

        for genre in genres:
            genre_df = df[df[genre_col] == genre]
            num_to_sample = min(images_per_genre + 10, len(genre_df))  # +10 for buffer
            sampled = genre_df.sample(n=num_to_sample, random_state=42)

            for idx, row in sampled.iterrows():
                if len(results) >= 600:
                    break

                # Save image to output directory
                img_data = row['image']
                img_filename = f"wikiart_{idx}_{genre}.jpg"
                img_path = os.path.join(WIKIART_OUTPUT_DIR, img_filename)

                try:
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        # Image stored as dictionary with bytes
                        img_bytes = img_data['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                        img.save(img_path)
                    elif isinstance(img_data, Image.Image):
                        img_data.save(img_path)
                    else:
                        # If it's raw bytes
                        with open(img_path, 'wb') as f:
                            f.write(img_data)

                    results.append({
                        "path": img_path,
                        "class": str(genre),
                        "class_name": str(genre)
                    })
                except Exception as e:
                    print(f"Error saving WikiArt image {idx}: {e}")
                    continue

            if len(results) >= 600:
                break
    else:
        # No genre column, just sample randomly
        sampled = df.sample(n=600, random_state=42)

        for idx, row in sampled.iterrows():
            img_data = row['image']
            img_filename = f"wikiart_{idx}.jpg"
            img_path = os.path.join(WIKIART_OUTPUT_DIR, img_filename)

            try:
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_bytes = img_data['bytes']
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)
                elif isinstance(img_data, Image.Image):
                    img_data.save(img_path)
                else:
                    with open(img_path, 'wb') as f:
                        f.write(img_data)

                results.append({
                    "path": img_path
                })
            except Exception as e:
                print(f"Error saving WikiArt image {idx}: {e}")
                continue

    print(f"Sampled {len(results)} WikiArt images")
    return results

def sample_food101():
    """Sample 1000 images from Food101 parquet files, evenly across 101 classes."""
    print("\n=== Sampling Food101 ===")
    parquet_dir = "/data/superstimuli_group/food101"
    ensure_dir(FOOD101_OUTPUT_DIR)

    # Read parquet files (prefer val/test split if available, otherwise train)
    parquet_files = list(Path(parquet_dir).glob("val-*.parquet"))
    if not parquet_files:
        parquet_files = list(Path(parquet_dir).glob("train-*.parquet"))

    print(f"Found {len(parquet_files)} Food101 parquet files")

    all_data = []
    for pf in parquet_files[:3]:  # Limit to first 3 files for speed
        df = pd.read_parquet(pf)
        all_data.append(df)

    df = pd.concat(all_data, ignore_index=True)
    print(f"Total Food101 images loaded: {len(df)}")
    print(f"Food101 columns: {df.columns.tolist()}")

    # Find label column
    label_col = None
    for col in ['label', 'category', 'class', 'fine_label']:
        if col in df.columns:
            label_col = col
            break

    results = []

    if label_col:
        categories = df[label_col].unique()
        print(f"Found {len(categories)} unique food categories")

        images_per_category = 1000 // len(categories)

        for category in categories:
            cat_df = df[df[label_col] == category]
            num_to_sample = min(images_per_category + 2, len(cat_df))
            sampled = cat_df.sample(n=num_to_sample, random_state=42)

            for idx, row in sampled.iterrows():
                if len(results) >= 1000:
                    break

                img_data = row['image']
                img_filename = f"food101_{idx}_{category}.jpg"
                img_path = os.path.join(FOOD101_OUTPUT_DIR, img_filename)

                try:
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                        img = Image.open(io.BytesIO(img_bytes))
                        img.save(img_path)
                    elif isinstance(img_data, Image.Image):
                        img_data.save(img_path)
                    else:
                        with open(img_path, 'wb') as f:
                            f.write(img_data)

                    results.append({
                        "path": img_path,
                        "class": str(category),
                        "class_name": str(category).replace('_', ' ')
                    })
                except Exception as e:
                    print(f"Error saving Food101 image {idx}: {e}")
                    continue

            if len(results) >= 1000:
                break
    else:
        # No label column found, sample randomly
        sampled = df.sample(n=1000, random_state=42)

        for idx, row in sampled.iterrows():
            img_data = row['image']
            img_filename = f"food101_{idx}.jpg"
            img_path = os.path.join(FOOD101_OUTPUT_DIR, img_filename)

            try:
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img_bytes = img_data['bytes']
                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(img_path)
                elif isinstance(img_data, Image.Image):
                    img_data.save(img_path)
                else:
                    with open(img_path, 'wb') as f:
                        f.write(img_data)

                results.append({
                    "path": img_path
                })
            except Exception as e:
                print(f"Error saving Food101 image {idx}: {e}")
                continue

    print(f"Sampled {len(results)} Food101 images")
    return results

def sample_beautiful_images():
    """Sample all beautiful images (169 images)."""
    print("\n=== Sampling Beautiful Images ===")
    base_path = "/data/superstimuli_group/beautiful_images"

    # Get all image files (excluding subdirectories)
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG']:
        image_files.extend(Path(base_path).glob(f'*{ext}'))

    image_files = [str(f) for f in image_files if f.is_file()]

    print(f"Found {len(image_files)} beautiful images")

    results = []
    for img_path in image_files:
        results.append({
            "path": img_path
        })

    return results

def sample_fractals():
    """Sample 150 images from Fractals dataset."""
    print("\n=== Sampling Fractals ===")
    base_path = "/data/superstimuli_group/fractals_and_fvis/fractals/images"

    image_files = get_image_files(base_path)
    print(f"Found {len(image_files)} fractal images")

    sampled = random.sample(image_files, min(150, len(image_files)))

    results = []
    for img_path in sampled:
        results.append({
            "path": img_path
        })

    return results

def sample_fvis():
    """Sample 150 images from Fvis dataset."""
    print("\n=== Sampling Fvis ===")
    base_path = "/data/superstimuli_group/fractals_and_fvis/first_layers_resized256_onevis/images"

    image_files = get_image_files(base_path)
    print(f"Found {len(image_files)} Fvis images")

    sampled = random.sample(image_files, min(150, len(image_files)))

    results = []
    for img_path in sampled:
        results.append({
            "path": img_path
        })

    return results

def main():
    """Main function to sample from all datasets and create JSONL file."""
    print("Starting image sampling process...")

    # Ensure output directory exists
    ensure_dir(os.path.dirname(OUTPUT_JSONL))

    all_samples = []

    # Sample from each dataset
    all_samples.extend(sample_imagenet_a())
    all_samples.extend(sample_imagenet_o())
    all_samples.extend(sample_imagenet_val())
    all_samples.extend(sample_species())
    all_samples.extend(sample_wikiart())
    all_samples.extend(sample_food101())
    all_samples.extend(sample_beautiful_images())
    all_samples.extend(sample_fractals())
    all_samples.extend(sample_fvis())

    # Write to JSONL file
    print(f"\n=== Writing {len(all_samples)} samples to {OUTPUT_JSONL} ===")

    with open(OUTPUT_JSONL, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Done! Created {OUTPUT_JSONL} with {len(all_samples)} images")

    # Print summary
    print("\n=== Summary ===")
    datasets = defaultdict(int)
    for sample in all_samples:
        path = sample['path']
        if 'imagenet-a' in path:
            datasets['ImageNet-A'] += 1
        elif 'imagenet-o' in path:
            datasets['ImageNet-O'] += 1
        elif 'imagenet_val' in path:
            datasets['ImageNet Val'] += 1
        elif 'species' in path:
            datasets['Species'] += 1
        elif 'wikiart' in path:
            datasets['WikiArt'] += 1
        elif 'food101' in path:
            datasets['Food101'] += 1
        elif 'beautiful_images' in path:
            datasets['Beautiful Images'] += 1
        elif 'first_layers' in path:
            datasets['Fvis'] += 1
        elif 'fractals' in path:
            datasets['Fractals'] += 1

    for dataset, count in sorted(datasets.items()):
        print(f"{dataset}: {count} images")
    print(f"Total: {sum(datasets.values())} images")

if __name__ == "__main__":
    main()
