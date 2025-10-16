#%%
# Compile best images:

# Sample 700 images from huggingface huggan/wikiart, roughly even across the classes of genre (11 classes) and style (16 classes)
# Sample 150 images from path fractals_and_fvis/first_layers_resized256_onevis
# Sample 150 images from path fractals_and_fvis/fractals
# Sample 400 images from ImageNet-A (path is imagenet-a), 2 per class
# Sample 1000 images from ImageNet-O (path is imagenet-o), 1 per class
# Sample 1000 images from ImageNet (path is /data/datasets/imagenet), 1 per class

# Collect them all into a single folder in current directory
# Create a json file that maps each image path to its source path and labels (if any)

# compile_images.py

import os
import shutil
import json
import random
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, Image

print("Starting image compilation script...")

# --- Configuration ---
# You can change these paths and numbers if needed.

# Hugging Face dataset
WIKIART_SAMPLES = 700

# Local datasets
FVIS_PATH = Path("fractals_and_fvis/first_layers_resized256_onevis")
FVIS_SAMPLES = 150

FRACTALS_PATH = Path("fractals_and_fvis/fractals")
FRACTALS_SAMPLES = 150

IMAGENET_A_PATH = Path("imagenet-a")
IMAGENET_A_SAMPLES_PER_CLASS = 2

IMAGENET_O_PATH = Path("imagenet-o")
IMAGENET_O_SAMPLES_PER_CLASS = 1

IMAGENET_PATH = Path("/data/datasets/imagenet")
IMAGENET_CLASSES_TO_SAMPLE = 1000 # Sample 1 image from 1000 random classes
IMAGENET_SAMPLES_PER_CLASS = 1

# Output
DEST_DIR = Path("compiled_images")
METADATA_FILE = Path("metadata.json")

# --- Helper Functions ---

def sample_wikiart(dest_dir, metadata):
    """Samples from huggan/wikiart with balanced classes."""
    print(f"\nSampling {WIKIART_SAMPLES} images from huggan/wikiart...")
    try:
        # Load dataset from Hugging Face Hub
        ds = load_dataset("huggan/wikiart", split="train")
        ds = ds.shuffle(seed=42)
    except Exception as e:
        print(f"Could not load huggan/wikiart dataset. Please check connection. Error: {e}")
        return

    # Get class names and calculate sampling targets
    genre_feature = ds.features['genre']
    style_feature = ds.features['style']
    num_genres = len(genre_feature.names)
    num_styles = len(style_feature.names)
    
    target_per_genre = WIKIART_SAMPLES / num_genres
    target_per_style = WIKIART_SAMPLES / num_styles

    genre_counts = {name: 0 for name in genre_feature.names}
    style_counts = {name: 0 for name in style_feature.names}
    
    selected_indices = []

    # Greedily select images to meet class targets
    for i in tqdm(range(len(ds)), desc="Balancing WikiArt classes"):
        if len(selected_indices) >= WIKIART_SAMPLES:
            break
        
        item = ds[i]
        genre = genre_feature.int2str(item['genre'])
        style = style_feature.int2str(item['style'])

        # Add if it helps balance either genre or style
        if genre_counts[genre] < target_per_genre or style_counts[style] < target_per_style:
            selected_indices.append(i)
            genre_counts[genre] += 1
            style_counts[style] += 1
            
    print(f"Selected {len(selected_indices)} images from WikiArt.")

    for idx in tqdm(selected_indices, desc="Saving WikiArt images"):
        item = ds[idx]
        img = item['image']
        genre = genre_feature.int2str(item['genre'])
        style = style_feature.int2str(item['style'])
        
        dest_filename = f"wikiart_{idx}.jpg"
        dest_path = dest_dir / dest_filename
        
        img.save(dest_path, "JPEG")
        
        metadata[str(dest_path)] = {
            "source_dataset": "huggan/wikiart",
            "source_index": idx,
            "labels": {
                "genre": genre,
                "style": style
            }
        }

def sample_local_flat(src_path, n_samples, prefix, dest_dir, metadata):
    """Randomly samples N images from a directory."""
    print(f"\nSampling {n_samples} images from {src_path}...")
    if not src_path.is_dir():
        print(f"Warning: Source directory not found: {src_path}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    try:
        all_images = [p for p in src_path.rglob('*') if p.suffix.lower() in image_extensions]
        if len(all_images) < n_samples:
            print(f"Warning: Found only {len(all_images)} images in {src_path}, taking all.")
            sampled_images = all_images
        else:
            sampled_images = random.sample(all_images, n_samples)
    except Exception as e:
        print(f"Error reading images from {src_path}: {e}")
        return

    for i, src_file in enumerate(tqdm(sampled_images, desc=f"Copying from {src_path.name}")):
        dest_filename = f"{prefix}_{i:04d}{src_file.suffix}"
        dest_path = dest_dir / dest_filename
        shutil.copy2(src_file, dest_path)
        
        metadata[str(dest_path)] = {
            "source_path": str(src_file),
            "labels": {}
        }
    print(f"Successfully copied {len(sampled_images)} images.")


def sample_local_per_class(src_path, samples_per_class, prefix, dest_dir, metadata, max_classes=None):
    """Samples a fixed number of images from each class subdirectory."""
    print(f"\nSampling {samples_per_class} image(s) per class from {src_path}...")
    if not src_path.is_dir():
        print(f"Warning: Source directory not found: {src_path}")
        return
        
    class_dirs = [d for d in src_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"Warning: No class subdirectories found in {src_path}.")
        return

    if max_classes and len(class_dirs) > max_classes:
        class_dirs = random.sample(class_dirs, max_classes)
        print(f"Randomly selected {max_classes} classes to sample from.")

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    image_count = 0
    
    for class_dir in tqdm(class_dirs, desc=f"Processing classes in {src_path.name}"):
        class_name = class_dir.name
        try:
            class_images = [p for p in class_dir.rglob('*') if p.suffix.lower() in image_extensions]
            
            if not class_images:
                continue

            if len(class_images) < samples_per_class:
                print(f"Warning: Class '{class_name}' has only {len(class_images)} images. Taking all.")
                sampled_images = class_images
            else:
                sampled_images = random.sample(class_images, samples_per_class)
            
            for i, src_file in enumerate(sampled_images):
                dest_filename = f"{prefix}_{class_name}_{i}{src_file.suffix}"
                dest_path = dest_dir / dest_filename
                shutil.copy2(src_file, dest_path)
                image_count += 1
                
                metadata[str(dest_path)] = {
                    "source_path": str(src_file),
                    "labels": {
                        "class": class_name
                    }
                }
        except Exception as e:
            print(f"Could not process directory {class_dir}. Error: {e}")

    print(f"Successfully copied {image_count} images.")


# --- Main Execution ---

def main():
    """Main function to run the image compilation process."""
    # Create destination directory
    DEST_DIR.mkdir(exist_ok=True)
    
    # Initialize metadata
    metadata = {}
    
    # 1. Sample from WikiArt
    sample_wikiart(DEST_DIR, metadata)
    
    # 2. Sample from FVIS
    sample_local_flat(FVIS_PATH, FVIS_SAMPLES, "fvis", DEST_DIR, metadata)
    
    # 3. Sample from Fractals
    sample_local_flat(FRACTALS_PATH, FRACTALS_SAMPLES, "fractal", DEST_DIR, metadata)
    
    # 4. Sample from ImageNet-A
    sample_local_per_class(IMAGENET_A_PATH, IMAGENET_A_SAMPLES_PER_CLASS, "imagenetA", DEST_DIR, metadata)

    # 5. Sample from ImageNet-O
    sample_local_per_class(IMAGENET_O_PATH, IMAGENET_O_SAMPLES_PER_CLASS, "imagenetO", DEST_DIR, metadata)

    # 6. Sample from ImageNet
    sample_local_per_class(IMAGENET_PATH, IMAGENET_SAMPLES_PER_CLASS, "imagenet", DEST_DIR, metadata, max_classes=IMAGENET_CLASSES_TO_SAMPLE)

    # Save metadata to JSON file
    print(f"\nSaving metadata for {len(metadata)} images to {METADATA_FILE}...")
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    total_images = len(list(DEST_DIR.glob('*')))
    print(f"\n✅ Done! Compiled a total of {total_images} images in '{DEST_DIR}'.")

if __name__ == "__main__":
    random.seed(42) # for reproducibility
    main()
# %%
