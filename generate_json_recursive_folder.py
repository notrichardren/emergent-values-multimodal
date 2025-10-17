#%%
import os
import json
import argparse
from typing import Iterable, List


SUPPORTED_EXTENSIONS: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")


def _collect_image_paths(root_dir: str) -> List[str]:
    """
    Walks ``root_dir`` recursively and returns a sorted list of absolute image paths.
    """
    image_paths: List[str] = []
    for current_dir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                abs_path = os.path.abspath(os.path.join(current_dir, filename))
                image_paths.append(abs_path)
    return sorted(image_paths)


def create_image_json_recursive(image_root: str, output_json_path: str) -> None:
    """
    Recursively scans ``image_root`` for images and writes them in the standard
    ``{"Images": [{"images": [<path>]}, ...]}`` format.
    """
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    image_paths = _collect_image_paths(image_root)
    if not image_paths:
        print(f"Warning: no images found under {image_root}")

    payload = {"Images": [{"images": [path]} for path in image_paths]}
    with open(output_json_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {len(image_paths)} images to {output_json_path}")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Recursively generate JSON listing all images beneath a root directory."
#     )
#     parser.add_argument("image_root", help="Root directory that contains image subfolders.")
#     parser.add_argument(
#         "output_json", help="Destination path for the generated JSON manifest."
#     )
#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

image_folder = "/data/superstimuli_group/politicians_large_256"
output_json = "/data/superstimuli_group/richard/small_runs/politicians_large_final.json"

create_image_json_recursive(image_folder, output_json)

# %%
