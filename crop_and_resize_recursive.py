#%%
import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

SUPPORTED_EXTENSIONS: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def _process_single_image(src_path: Path, dst_path: Path, size: Tuple[int, int]) -> None:
    with Image.open(src_path) as img:
        width, height = img.size
        short_side = min(width, height)
        left = (width - short_side) / 2
        top = (height - short_side) / 2
        right = (width + short_side) / 2
        bottom = (height + short_side) / 2
        cropped = img.crop((left, top, right, bottom))
        resized = cropped.resize(size, Image.Resampling.BICUBIC)
        if resized.mode in ("RGBA", "P"):
            resized = resized.convert("RGB")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst_path, "jpeg", quality=95)


def process_images_recursively(input_root: str, output_root: str, size: Tuple[int, int] = (256, 256)) -> None:
    src_root = Path(input_root).resolve()
    dst_root = Path(output_root).resolve()

    if not src_root.is_dir():
        raise FileNotFoundError(f"Input directory '{src_root}' not found.")

    num_processed = 0
    for dirpath, _, filenames in os.walk(src_root):
        for filename in filenames:
            if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            rel_dir = Path(dirpath).relative_to(src_root)
            src_path = Path(dirpath) / filename
            dst_filename = f"{Path(filename).stem}_cropped.jpg"
            dst_path = dst_root / rel_dir / dst_filename
            try:
                _process_single_image(src_path, dst_path, size)
                num_processed += 1
            except Exception as exc:
                print(f"Error processing {src_path}: {exc}")

    print(f"Processed {num_processed} images into {dst_root}")


if __name__ == "__main__":
    INPUT_ROOT = "/data/superstimuli_group/politicians_large"
    OUTPUT_ROOT = "/data/superstimuli_group/politicians_large_256"
    TARGET_SIZE = (256, 256)

    process_images_recursively(INPUT_ROOT, OUTPUT_ROOT, TARGET_SIZE)


# %%
