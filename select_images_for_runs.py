#!/usr/bin/env python3
"""
Script to randomly sample images from various datasets for superstimuli experiments.
Each dataset is sampled evenly across classes/genres and output to its own JSONL file.
"""

# We are currently in /data/superstimuli_group/select_images_for_runs/select_images_for_runs.py

# I want you to randomly sample images from the following places:

# ImageNet-A (200 classes; 7.5k images)
# - path: /data/superstimuli_group/imagenet-a
# - Please sample evenly across 200 classes to get 2000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/imagenet-a/.../...) as well as each "class" being mapped. Also make it so there's also a column called "class_name" (if class is not interpretable or it's some 2143hjk looking thing)
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]

# ImageNet-O (1000 classes; 2k images)
# - path: /data/superstimuli_group/imagenet-o
# - Please sample evenly across 1000 classes to get 6000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/imagenet-o/.../...) as well as each "class" being mapped. Also make it so there's also a column called "class_name" (if class is not interpretable or it's some 2143hjk looking thing)
# Then we will select top 3000 [do not worry about this, we will be the ones doing the selecting]

# WikiArt (11 genre classes; 11k images)
# - path: /data/superstimuli_group/wikiart (but it's a huggingface datasets parquet thing)
# - Please sample roughly evenly across genres and get around 6000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/wikiart/.../...) as well as each "genre" being mapped
# Then we will select top 3000 [do not worry about this, we will be the ones doing the selecting]

# ImageNet_Val (1000 classes; 50k images)
# - path: /data/superstimuli_group/imagenet_val
# - Please sample evenly across 1000 classes to get 6000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/imagenet_val/.../...) as well as each "genre" being mapped
# Then we will select top 3000 [do not worry about this, we will be the ones doing the selecting]

# Food101 (101 categories; 101k images)
# - path: /data/superstimuli_group/food101 (but it's a huggingface datasets parquet thing)
# - Please sample evenly across 101 classes to get ~4000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/food101/.../...) as well as each "class" being mapped.
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]

# Species (640 categories)
# - path: /data/superstimuli_group/species_data/species_for_finegrained/species_splits/test/iid
# - Please sample evenly across 640 classes to get ~4000 images
# - Please put this in a its own jsonl file where each json maps "path":path (full path, so /data/superstimuli_group/species_data/.../...) as well as each "class" being mapped.
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]

# Coco (big folder; just randomly sample)
# - path: /data/superstimuli_group/coco/train2017
# - Please sample to get ~4000 images
# - Its own jsonl with path
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]

# Beautiful images (169 images)
# - path: /data/superstimuli_group/beautiful_images
# - Please just include all images
# - Its own jsonl with path
# Then we will select top 150 [do not worry about this, we will be the ones doing the selecting]

# Fractals (14248 images)
# - path: /data/superstimuli_group/beautiful_images
# - Please sample to get 4000 images
# - Its own jsonl with path
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]

# Fvis (4690 images total)
# - Please sample to get 4000 images
# - Its own jsonl with path
# Then we will select top 2000 [do not worry about this, we will be the ones doing the selecting]


import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import glob

# Set random seed for reproducibility
random.seed(42)

BASE_DIR = Path("/data/superstimuli_group")
OUTPUT_DIR = Path("/data/superstimuli_group/select_images_for_runs")

# ImageNet class names mapping (synset to human-readable)
IMAGENET_CLASS_NAMES = {}  # Will be populated if needed


def get_image_files(directory: Path, extensions=('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')) -> List[Path]:
    """Get all image files in a directory."""
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'**/*{ext}'))
    return image_files


def sample_evenly_by_class(base_path: Path, num_samples: int, class_key: str = "class") -> List[Dict]:
    """
    Sample images evenly across classes in a directory structure where each subdirectory is a class.
    """
    # Get all class directories
    class_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"Warning: No class directories found in {base_path}")
        return []

    # Group images by class
    images_by_class = {}
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = get_image_files(class_dir)
        if images:
            images_by_class[class_name] = images

    if not images_by_class:
        print(f"Warning: No images found in {base_path}")
        return []

    # Calculate samples per class
    num_classes = len(images_by_class)
    samples_per_class = num_samples // num_classes
    extra_samples = num_samples % num_classes

    print(f"  Found {num_classes} classes")
    print(f"  Sampling {samples_per_class} images per class (plus {extra_samples} extra)")

    # Sample from each class
    sampled = []
    for class_name, images in sorted(images_by_class.items()):
        # Determine how many samples for this class
        n_samples = samples_per_class
        if extra_samples > 0:
            n_samples += 1
            extra_samples -= 1

        # Sample (with replacement if needed)
        if len(images) >= n_samples:
            selected = random.sample(images, n_samples)
        else:
            selected = random.choices(images, k=n_samples)

        for img_path in selected:
            sampled.append({
                "path": str(img_path.absolute()),
                class_key: class_name,
                "class_name": class_name  # For readability
            })

    return sampled


def sample_imagenet_a():
    """Sample 2000 images evenly across 200 classes from ImageNet-A."""
    print("\n=== Sampling ImageNet-A ===")
    base_path = BASE_DIR / "imagenet-a" / "imagenet-a"
    sampled = sample_evenly_by_class(base_path, 2000, "class")

    output_file = OUTPUT_DIR / "imagenet_a_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_imagenet_o():
    """Sample 6000 images evenly across classes from ImageNet-O."""
    print("\n=== Sampling ImageNet-O ===")
    base_path = BASE_DIR / "imagenet-o" / "imagenet-o"
    sampled = sample_evenly_by_class(base_path, 6000, "class")

    output_file = OUTPUT_DIR / "imagenet_o_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_wikiart():
    """Sample 6000 images evenly across genres from WikiArt HuggingFace dataset."""
    print("\n=== Sampling WikiArt ===")

    try:
        import pandas as pd
        import pyarrow.parquet as pq
        from PIL import Image
        import io
    except ImportError:
        print("  Error: pandas, pyarrow, and PIL required for reading parquet files")
        return []

    # Create output directory for images
    wikiart_img_dir = OUTPUT_DIR / "wikiart_samples"
    wikiart_img_dir.mkdir(parents=True, exist_ok=True)

    # Load all parquet files
    parquet_dir = BASE_DIR / "wikiart"
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))

    print(f"  Found {len(parquet_files)} parquet files")

    # Read all data
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total images in dataset: {len(df)}")

    # Check what columns are available
    print(f"  Columns: {df.columns.tolist()}")

    # Find genre column (might be 'genre', 'style', or similar)
    genre_col = None
    for col in ['genre', 'style', 'artist', 'Genre', 'Style']:
        if col in df.columns:
            genre_col = col
            break

    if genre_col is None:
        print(f"  Warning: No genre column found. Using random sampling.")
        sampled_df = df.sample(n=min(6000, len(df)), random_state=42)
        genre_col = 'unknown'
    else:
        print(f"  Using genre column: {genre_col}")
        # Sample evenly by genre
        genres = df[genre_col].unique()
        print(f"  Found {len(genres)} genres: {genres[:11]}")

        samples_per_genre = 6000 // len(genres)
        extra = 6000 % len(genres)

        sampled_dfs = []
        for i, genre in enumerate(sorted(genres)):
            genre_df = df[df[genre_col] == genre]
            n_samples = samples_per_genre + (1 if i < extra else 0)
            n_samples = min(n_samples, len(genre_df))
            sampled_genre = genre_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled_genre)

        sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # Save images to disk
    print(f"  Extracting and saving {len(sampled_df)} images to disk...")
    sampled = []
    for idx, row in sampled_df.iterrows():
        try:
            # Get image from parquet (usually in 'image' column as PIL Image or bytes)
            if 'image' in row:
                img = row['image']

                # Handle different image formats
                if isinstance(img, dict) and 'bytes' in img:
                    # HuggingFace format with bytes
                    img = Image.open(io.BytesIO(img['bytes']))
                elif hasattr(img, 'save'):
                    # Already a PIL Image
                    pass
                else:
                    print(f"  Warning: Unexpected image format for row {idx}")
                    continue

                # Save image
                genre_name = str(row[genre_col]).replace('/', '_').replace(' ', '_')
                img_filename = f"{genre_name}_{idx}.jpg"
                img_path = wikiart_img_dir / img_filename
                img.save(img_path, 'JPEG')

                sampled.append({
                    "path": str(img_path.absolute()),
                    "genre": str(row[genre_col]),
                })
            else:
                print(f"  Warning: No image column found in row {idx}")
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue

    output_file = OUTPUT_DIR / "wikiart_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_imagenet_val():
    """Sample 6000 images evenly across 1000 classes from ImageNet Val."""
    print("\n=== Sampling ImageNet Val ===")
    base_path = BASE_DIR / "imagenet_val"
    sampled = sample_evenly_by_class(base_path, 6000, "class")

    output_file = OUTPUT_DIR / "imagenet_val_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_food101():
    """Sample 4000 images evenly across 101 classes from Food101 HuggingFace dataset."""
    print("\n=== Sampling Food101 ===")

    try:
        import pandas as pd
        from PIL import Image
        import io
    except ImportError:
        print("  Error: pandas and PIL required for reading parquet files")
        return []

    # Create output directory for images
    food101_img_dir = OUTPUT_DIR / "food101_samples"
    food101_img_dir.mkdir(parents=True, exist_ok=True)

    # Load all parquet files
    parquet_dir = BASE_DIR / "food101"
    parquet_files = sorted(parquet_dir.glob("train-*.parquet"))

    print(f"  Found {len(parquet_files)} parquet files")

    # Read all data
    dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total images in dataset: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")

    # Find class column
    class_col = None
    for col in ['label', 'class', 'category', 'food', 'Label']:
        if col in df.columns:
            class_col = col
            break

    if class_col is None:
        print(f"  Warning: No class column found. Using random sampling.")
        sampled_df = df.sample(n=min(4000, len(df)), random_state=42)
        class_col = 'unknown'
    else:
        print(f"  Using class column: {class_col}")
        # Sample evenly by class
        classes = df[class_col].unique()
        print(f"  Found {len(classes)} classes")

        samples_per_class = 4000 // len(classes)
        extra = 4000 % len(classes)

        sampled_dfs = []
        for i, cls in enumerate(sorted(classes)):
            cls_df = df[df[class_col] == cls]
            n_samples = samples_per_class + (1 if i < extra else 0)
            n_samples = min(n_samples, len(cls_df))
            sampled_cls = cls_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled_cls)

        sampled_df = pd.concat(sampled_dfs, ignore_index=True)

    # Save images to disk
    print(f"  Extracting and saving {len(sampled_df)} images to disk...")
    sampled = []
    for idx, row in sampled_df.iterrows():
        try:
            # Get image from parquet (usually in 'image' column as PIL Image or bytes)
            if 'image' in row:
                img = row['image']

                # Handle different image formats
                if isinstance(img, dict) and 'bytes' in img:
                    # HuggingFace format with bytes
                    img = Image.open(io.BytesIO(img['bytes']))
                elif hasattr(img, 'save'):
                    # Already a PIL Image
                    pass
                else:
                    print(f"  Warning: Unexpected image format for row {idx}")
                    continue

                # Save image
                class_name = str(row[class_col]).replace('/', '_').replace(' ', '_')
                img_filename = f"{class_name}_{idx}.jpg"
                img_path = food101_img_dir / img_filename
                img.save(img_path, 'JPEG')

                sampled.append({
                    "path": str(img_path.absolute()),
                    "class": str(row[class_col]),
                    "class_name": str(row[class_col]),
                })
            else:
                print(f"  Warning: No image column found in row {idx}")
        except Exception as e:
            print(f"  Error processing row {idx}: {e}")
            continue

    output_file = OUTPUT_DIR / "food101_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_species():
    """Sample 4000 images evenly across 640 classes from Species dataset."""
    print("\n=== Sampling Species ===")
    base_path = BASE_DIR / "species_data" / "species_for_finegrained" / "species_splits" / "test" / "iid"
    sampled = sample_evenly_by_class(base_path, 4000, "class")

    output_file = OUTPUT_DIR / "species_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_coco():
    """Sample 4000 images randomly from COCO train2017."""
    print("\n=== Sampling COCO ===")
    base_path = BASE_DIR / "coco" / "train2017"

    all_images = get_image_files(base_path)
    print(f"  Found {len(all_images)} total images")

    # Random sample
    n_samples = min(4000, len(all_images))
    sampled_images = random.sample(all_images, n_samples)

    sampled = []
    for img_path in sampled_images:
        sampled.append({
            "path": str(img_path.absolute())
        })

    output_file = OUTPUT_DIR / "coco_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_beautiful_images():
    """Include all images from beautiful_images dataset."""
    print("\n=== Sampling Beautiful Images ===")
    base_path = BASE_DIR / "beautiful_images"

    all_images = get_image_files(base_path)
    print(f"  Found {len(all_images)} total images")

    sampled = []
    for img_path in all_images:
        sampled.append({
            "path": str(img_path.absolute())
        })

    output_file = OUTPUT_DIR / "beautiful_images_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_fractals():
    """Sample 4000 images randomly from fractals dataset."""
    print("\n=== Sampling Fractals ===")
    base_path = BASE_DIR / "fractals_and_fvis" / "fractals"

    all_images = get_image_files(base_path)
    print(f"  Found {len(all_images)} total images")

    # Random sample
    n_samples = min(4000, len(all_images))
    sampled_images = random.sample(all_images, n_samples)

    sampled = []
    for img_path in sampled_images:
        sampled.append({
            "path": str(img_path.absolute())
        })

    output_file = OUTPUT_DIR / "fractals_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def sample_fvis():
    """Sample 4000 images randomly from fvis dataset."""
    print("\n=== Sampling Fvis ===")
    base_path = BASE_DIR / "fractals_and_fvis" / "first_layers_resized256_onevis"

    all_images = get_image_files(base_path)
    print(f"  Found {len(all_images)} total images")

    # Random sample
    n_samples = min(4000, len(all_images))
    sampled_images = random.sample(all_images, n_samples)

    sampled = []
    for img_path in sampled_images:
        sampled.append({
            "path": str(img_path.absolute())
        })

    output_file = OUTPUT_DIR / "fvis_samples.jsonl"
    with open(output_file, 'w') as f:
        for item in sampled:
            f.write(json.dumps(item) + '\n')

    print(f"  Saved {len(sampled)} samples to {output_file}")
    return sampled


def main():
    """Run all sampling functions."""
    print("=" * 60)
    print("Starting image sampling for superstimuli experiments")
    print("=" * 60)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all sampling functions
    results = {}

    try:
        results['imagenet_a'] = sample_imagenet_a()
    except Exception as e:
        print(f"  Error sampling ImageNet-A: {e}")
        results['imagenet_a'] = []

    try:
        results['imagenet_o'] = sample_imagenet_o()
    except Exception as e:
        print(f"  Error sampling ImageNet-O: {e}")
        results['imagenet_o'] = []

    try:
        results['wikiart'] = sample_wikiart()
    except Exception as e:
        print(f"  Error sampling WikiArt: {e}")
        results['wikiart'] = []

    try:
        results['imagenet_val'] = sample_imagenet_val()
    except Exception as e:
        print(f"  Error sampling ImageNet Val: {e}")
        results['imagenet_val'] = []

    try:
        results['food101'] = sample_food101()
    except Exception as e:
        print(f"  Error sampling Food101: {e}")
        results['food101'] = []

    try:
        results['species'] = sample_species()
    except Exception as e:
        print(f"  Error sampling Species: {e}")
        results['species'] = []

    try:
        results['coco'] = sample_coco()
    except Exception as e:
        print(f"  Error sampling COCO: {e}")
        results['coco'] = []

    try:
        results['beautiful_images'] = sample_beautiful_images()
    except Exception as e:
        print(f"  Error sampling Beautiful Images: {e}")
        results['beautiful_images'] = []

    try:
        results['fractals'] = sample_fractals()
    except Exception as e:
        print(f"  Error sampling Fractals: {e}")
        results['fractals'] = []

    try:
        results['fvis'] = sample_fvis()
    except Exception as e:
        print(f"  Error sampling Fvis: {e}")
        results['fvis'] = []

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for dataset, samples in results.items():
        print(f"  {dataset:20s}: {len(samples):5d} samples")
    print(f"  {'TOTAL':20s}: {sum(len(s) for s in results.values()):5d} samples")
    print("=" * 60)
    print(f"\nAll output files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
