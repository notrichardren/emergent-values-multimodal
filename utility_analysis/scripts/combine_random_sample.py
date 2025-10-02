import json
import random
from pathlib import Path

# Absolute paths per user preference
AI_GEN_IMG_PATH = Path("/data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/ai_gen_img.json")
WIKIART_STYLE_PATH = Path("/data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/wikiart_style.json")
MULTIMODAL_PATH = Path("/data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/multimodal_options_hierarchical.json")
OUTPUT_PATH = Path("/data/wenjie_jacky_mo/emergent-values/utility_analysis/shared_options/combined_set.json")

SAMPLE_SIZE = 150


def load_images(path: Path):
    with path.open("r") as f:
        data = json.load(f)
    images = data.get("Images", [])
    return images


def sample_images(images, k: int):
    if not images:
        return []
    k_eff = min(k, len(images))
    return random.sample(images, k=k_eff)


def main():
    ai_gen_images = load_images(AI_GEN_IMG_PATH)
    wikiart_images = load_images(WIKIART_STYLE_PATH)
    multimodal_images = load_images(MULTIMODAL_PATH)

    sampled_ai = sample_images(ai_gen_images, SAMPLE_SIZE)
    sampled_wiki = sample_images(wikiart_images, SAMPLE_SIZE)
    sampled_multi = sample_images(multimodal_images, SAMPLE_SIZE)

    combined = sampled_ai + sampled_wiki + sampled_multi

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump({"Images": combined}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main() 