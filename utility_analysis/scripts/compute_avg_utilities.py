#!/usr/bin/env python3

import re
from collections import defaultdict
from typing import Dict, Tuple, List
import argparse
import os

AGE_SUMMARY_PATH = "/data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_age/summary_qwen25-vl-72b-instruct.txt"
GENDER_SUMMARY_PATH = "/data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_gender/summary_qwen25-vl-72b-instruct.txt"
RACE_SUMMARY_PATH = "/data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/72b_race/summary_qwen25-vl-72b-instruct.txt"

UTILITY_LINE_REGEX = re.compile(r"^/.+?:\s+mean=([+-]?\d+(?:\.\d+)?),")


def parse_category_and_mean(line: str) -> Tuple[str, float]:
    """Extract category (folder before filename) and mean value from a summary line."""
    # Example line:
    # /data/.../fair_face_age/3_9/fairface_000641.jpg: mean=2.3230, variance=0.0597
    if ": mean=" not in line:
        raise ValueError("Line does not contain a mean value")

    path_part = line.split(":", 1)[0].strip()
    path_segments = path_part.split("/")
    if len(path_segments) < 2:
        raise ValueError("Unexpected path format")
    category = path_segments[-2]  # folder right before the filename

    match = UTILITY_LINE_REGEX.match(line.strip())
    if not match:
        raise ValueError("Could not parse mean value")
    mean_value = float(match.group(1))

    return category, mean_value


def compute_category_averages(summary_path: str) -> List[Tuple[str, float, int]]:
    """Read a summary file and compute average mean utility for each category.

    Returns a list of tuples: (category, average_mean, count).
    """
    category_to_sum: Dict[str, float] = defaultdict(float)
    category_to_count: Dict[str, int] = defaultdict(int)

    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if ": mean=" not in line:
                continue
            try:
                category, mean_value = parse_category_and_mean(line)
            except ValueError:
                continue
            category_to_sum[category] += mean_value
            category_to_count[category] += 1

    results: List[Tuple[str, float, int]] = []
    for category, total in category_to_sum.items():
        count = category_to_count[category]
        average = total / count if count > 0 else 0.0
        results.append((category, average, count))

    return results


def print_section(title: str, summary_path: str, ascending: bool) -> None:
    order = "asc" if ascending else "desc"
    print(f"\n{title} (sorted by average utility, {order})")
    print("-" * (len(title) + len(" (sorted by average utility, ") + len(order) + 1))
    results = compute_category_averages(summary_path)
    results.sort(key=lambda x: x[1], reverse=not ascending)
    for category, avg, count in results:
        print(f"{category}: avg_mean={avg:.4f} (n={count})")


def infer_default_title(summary_path: str) -> str:
    parent_dir = os.path.basename(os.path.dirname(summary_path))
    return f"Averages for {parent_dir} by category"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and print average utilities per category.")
    parser.add_argument("--ascending", action="store_true", help="Sort ascending by average utility (default: descending)")
    parser.add_argument("--file", action="append", default=[], help="Extra summary file to process (repeatable)")
    parser.add_argument("--name", action="append", default=[], help="Optional custom title for each --file, in order")
    parser.add_argument("--skip-defaults", action="store_true", help="Skip the built-in age/gender/race sections")
    args = parser.parse_args()

    if not args.skip_defaults:
        print_section("Age averages by category", AGE_SUMMARY_PATH, args.ascending)
        print_section("Gender averages by category", GENDER_SUMMARY_PATH, args.ascending)
        print_section("Race averages by category", RACE_SUMMARY_PATH, args.ascending)

    # Process extra files, if any
    for idx, path in enumerate(args.file):
        title = args.name[idx] if idx < len(args.name) else infer_default_title(path)
        print_section(title, path, args.ascending) 