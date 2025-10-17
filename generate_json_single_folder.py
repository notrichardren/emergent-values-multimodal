#%%
import os
import json
import argparse

def create_image_json(image_folder, output_json_path):
    """
    Scans a folder for images and generates a JSON file in the specified format.

    Args:
        image_folder (str): The path to the folder containing image files.
        output_json_path (str): The path where the output JSON file will be saved.
    """
    # First, check if the provided image folder path exists.
    if not os.path.isdir(image_folder):
        print(f"❌ Error: The specified folder does not exist: {image_folder}")
        return

    # Define the common image file extensions to look for.
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

    image_data_list = []
    # Use sorted() to ensure the file order is consistent every time.
    for filename in sorted(os.listdir(image_folder)):
        # Check if the file has a supported image extension.
        if filename.lower().endswith(supported_extensions):
            # Create the full, absolute path for the image file.
            full_path = os.path.abspath(os.path.join(image_folder, filename))
            
            # Format the data into the required dictionary structure.
            image_entry = {
                "images": [full_path]
            }
            image_data_list.append(image_entry)

    if not image_data_list:
        print(f"⚠️ Warning: No images found in the folder: {image_folder}")

    # Create the final JSON structure with the top-level "Images" key.
    final_json_structure = {
        "Images": image_data_list
    }

    # Write the data to the specified output JSON file.
    try:
        with open(output_json_path, 'w') as json_file:
            # Use indent=2 to make the JSON file readable, like your example.
            json.dump(final_json_structure, json_file, indent=2)
        print(f"✅ Success! Generated JSON file at: {output_json_path}")
    except IOError as e:
        print(f"❌ Error writing to file {output_json_path}: {e}")

# def main():
#     """
#     Parses command-line arguments and runs the JSON generation script.
#     """
#     parser = argparse.ArgumentParser(
#         description="Generate a JSON file from a folder of images in a specific format.",
#         formatter_class=argparse.RawTextHelpFormatter
#     )
    
#     parser.add_argument(
#         "image_folder",
#         type=str,
#         help="Path to the folder containing your image files."
#     )
    
#     parser.add_argument(
#         "output_json",
#         type=str,
#         help="Full path for the output .json file (e.g., /path/to/output.json)."
#     )

    # args = parser.parse_args()

image_folder = "/data/superstimuli_group/politicians_large"
output_json = "big_runs/final_images.json"

create_image_json(image_folder, output_json)

# if __name__ == "__main__":
    # main()
# %%
