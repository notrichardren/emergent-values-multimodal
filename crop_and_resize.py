#%%
import os
from PIL import Image

def process_images_in_directory(input_dir, output_dir='cropped', size=(256, 256)):
    """
    Finds all images in an input directory, center-crops them to a square,
    and resizes them, saving the results in an output directory.

    Args:
        input_dir (str): The path to the folder containing source images.
        output_dir (str): The path to the folder where cropped images will be saved.
        size (tuple): The target (width, height) for the final images.
    """
    # Check if the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of supported image file extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    print(f"Scanning '{input_dir}' for images...")

    for filename in os.listdir(input_dir):
        # Process only files with supported image extensions
        if filename.lower().endswith(supported_extensions):
            try:
                # Construct full file paths
                input_path = os.path.join(input_dir, filename)
                img = Image.open(input_path)

                # --- Center Crop Logic ---
                width, height = img.size
                short_side = min(width, height)
                left = (width - short_side) / 2
                top = (height - short_side) / 2
                right = (width + short_side) / 2
                bottom = (height + short_side) / 2
                img_cropped = img.crop((left, top, right, bottom))
                
                # --- Resize Logic ---
                # Bicubic resampling is a high-quality method for scaling images.
                img_resized = img_cropped.resize(size, Image.Resampling.BICUBIC)

                # --- Save Logic ---
                new_filename = f"{os.path.splitext(filename)[0]}_cropped.jpg"
                save_path = os.path.join(output_dir, new_filename)
                
                # Convert to RGB to handle PNG transparency and ensure JPEG compatibility
                if img_resized.mode in ("RGBA", "P"):
                    img_resized = img_resized.convert("RGB")
                    
                img_resized.save(save_path, 'jpeg', quality=95)
                print(f"  -> Processed {filename}")

            except Exception as e:
                print(f"  -> Error processing {filename}: {e}")

# --- HOW TO USE ---

# 1. Define the name of the folder where your original images are.
#    Create this folder in the same directory as your script and put your images inside it.
INPUT_FOLDER = 'run_big/intermediate_images' 

# 2. Define the name of the folder for the results.
OUTPUT_FOLDER = 'run_small/final_images'

# 3. Run the function.
process_images_in_directory(INPUT_FOLDER, OUTPUT_FOLDER)

print(f"\n✅ Processing complete! Check the '{OUTPUT_FOLDER}' folder.")
# %%
