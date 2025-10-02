#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Use a non-interactive backend for servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ImageUtility:
    image_path: str
    style: str
    mean_utility: float
    variance: float


@dataclass
class StyleAggregates:
    style: str
    num_images: int
    mean_of_means: float
    median_of_means: float
    standard_error: float
    ci95_low: float
    ci95_high: float
    weighted_mean: float
    weighted_standard_error: float
    weighted_ci95_low: float
    weighted_ci95_high: float


STYLE_LINE_PATTERN = re.compile(
    r"^(.+?/wikiart_by_style/([^/]+)/[^:]+):\s*mean=([+-]?(?:\d+\.?\d*|\.\d+)),\s*variance=([+-]?(?:\d+\.?\d*|\.\d+))\s*$"
)


def parse_summary_file(summary_path: str) -> List[ImageUtility]:
    image_utilities: List[ImageUtility] = []
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as file:
        for line in file:
            line_stripped = line.strip()
            match = STYLE_LINE_PATTERN.match(line_stripped)
            if not match:
                continue
            full_path, style, mean_str, var_str = match.groups()
            try:
                mean_val = float(mean_str)
                var_val = float(var_str)
            except ValueError:
                continue
            # Guard against negative or zero variance for weighting later
            if not math.isfinite(var_val) or var_val < 0:
                var_val = float("nan")
            image_utilities.append(
                ImageUtility(
                    image_path=full_path,
                    style=style,
                    mean_utility=mean_val,
                    variance=var_val,
                )
            )
    if len(image_utilities) == 0:
        raise ValueError(
            "No image utility lines were parsed. Ensure the file format matches the expected pattern."
        )
    return image_utilities


def compute_style_aggregates(image_utilities: List[ImageUtility], weight_epsilon: float = 1e-3) -> Tuple[List[StyleAggregates], Dict[str, List[float]]]:
    style_to_means: Dict[str, List[float]] = defaultdict(list)
    style_to_vars: Dict[str, List[float]] = defaultdict(list)

    for item in image_utilities:
        style_to_means[item.style].append(item.mean_utility)
        style_to_vars[item.style].append(item.variance)

    aggregates: List[StyleAggregates] = []

    for style, means in style_to_means.items():
        variances = style_to_vars[style]
        values = np.array(means, dtype=float)
        n = len(values)

        # Unweighted mean/median
        mean_of_means = float(np.mean(values)) if n > 0 else float("nan")
        median_of_means = float(np.median(values)) if n > 0 else float("nan")

        # Standard error of the mean (sample-based)
        if n > 1:
            std_dev = float(np.std(values, ddof=1))
            standard_error = std_dev / math.sqrt(n)
        else:
            std_dev = float("nan")
            standard_error = float("nan")

        if math.isfinite(standard_error):
            ci95_low = mean_of_means - 1.96 * standard_error
            ci95_high = mean_of_means + 1.96 * standard_error
        else:
            ci95_low = float("nan")
            ci95_high = float("nan")

        # Precision-weighted mean using 1/(variance + epsilon)
        weights = []
        for var in variances:
            if not math.isfinite(var) or var <= 0:
                # Fallback to epsilon to avoid infinite/negative weight
                w = 1.0 / weight_epsilon
            else:
                w = 1.0 / (var + weight_epsilon)
            weights.append(w)
        weights_arr = np.array(weights, dtype=float)

        if np.sum(weights_arr) > 0:
            weighted_mean = float(np.average(values, weights=weights_arr))
            # For independent estimates with known variances, se ~ sqrt(1/sum(w))
            weighted_standard_error = float(math.sqrt(1.0 / float(np.sum(weights_arr))))
            weighted_ci95_low = weighted_mean - 1.96 * weighted_standard_error
            weighted_ci95_high = weighted_mean + 1.96 * weighted_standard_error
        else:
            weighted_mean = float("nan")
            weighted_standard_error = float("nan")
            weighted_ci95_low = float("nan")
            weighted_ci95_high = float("nan")

        aggregates.append(
            StyleAggregates(
                style=style,
                num_images=n,
                mean_of_means=mean_of_means,
                median_of_means=median_of_means,
                standard_error=standard_error,
                ci95_low=ci95_low,
                ci95_high=ci95_high,
                weighted_mean=weighted_mean,
                weighted_standard_error=weighted_standard_error,
                weighted_ci95_low=weighted_ci95_low,
                weighted_ci95_high=weighted_ci95_high,
            )
        )

    # Sort by unweighted mean desc by default
    aggregates.sort(key=lambda a: a.mean_of_means, reverse=True)
    return aggregates, style_to_means


def save_csv(aggregates: List[StyleAggregates], csv_path: str) -> None:
    header = (
        "style,num_images,mean,median,se,ci95_low,ci95_high,"
        "weighted_mean,weighted_se,weighted_ci95_low,weighted_ci95_high\n"
    )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        for ag in aggregates:
            f.write(
                f"{ag.style},{ag.num_images},{ag.mean_of_means:.6f},{ag.median_of_means:.6f},"
                f"{_fmt_or_nan(ag.standard_error)},{_fmt_or_nan(ag.ci95_low)},"
                f"{_fmt_or_nan(ag.ci95_high)},{_fmt_or_nan(ag.weighted_mean)},"
                f"{_fmt_or_nan(ag.weighted_standard_error)},{_fmt_or_nan(ag.weighted_ci95_low)},"
                f"{_fmt_or_nan(ag.weighted_ci95_high)}\n"
            )


def _fmt_or_nan(value: float) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def _prettify_style(style: str) -> str:
    return style.replace("_", " ")


def plot_bar_with_ci(aggregates: List[StyleAggregates], out_path: str, use_weighted: bool = False) -> None:
    styles = [ag.style for ag in aggregates]
    labels = [_prettify_style(s) for s in styles]

    if use_weighted:
        means = [ag.weighted_mean for ag in aggregates]
        ses = [ag.weighted_standard_error for ag in aggregates]
        title = "Style utility (precision-weighted mean ± 95% CI)"
        fname = os.path.join(out_path, "style_utility_bar_weighted.png")
    else:
        means = [ag.mean_of_means for ag in aggregates]
        ses = [ag.standard_error for ag in aggregates]
        title = "Style utility (mean ± 95% CI)"
        fname = os.path.join(out_path, "style_utility_bar.png")

    # Compute 95% CI lengths from se
    yerr = []
    for se in ses:
        if se is None or not math.isfinite(se):
            yerr.append([0.0, 0.0])
        else:
            ci = 1.96 * se
            yerr.append([ci, ci])

    num_styles = len(styles)
    width = max(10.0, min(2.0 * num_styles, 60.0))
    height = 6.0

    fig, ax = plt.subplots(figsize=(width, height))
    x = np.arange(num_styles)

    ax.bar(x, means, yerr=np.array(yerr).T, capsize=3, color="#4C78A8", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Utility")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)


def plot_boxplot(style_to_means: Dict[str, List[float]], aggregates: List[StyleAggregates], out_dir: str) -> None:
    # Order styles by unweighted mean desc to align with bar plot
    ordered_styles = [ag.style for ag in aggregates]
    labels = [_prettify_style(s) for s in ordered_styles]
    data = [style_to_means[s] for s in ordered_styles]

    num_styles = len(ordered_styles)
    width = max(10.0, min(2.0 * num_styles, 60.0))
    height = 8.0

    fig, ax = plt.subplots(figsize=(width, height))
    ax.boxplot(data, showfliers=False, notch=False)
    ax.set_xticks(np.arange(1, num_styles + 1))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("Utility (per image)")
    ax.set_title("Distribution of utilities by style (boxplot)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "style_utility_box.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate image utility by style and visualize.")
    parser.add_argument(
        "--summary",
        type=str,
        default="/data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/wiki_subset_72b/summary_qwen25-vl-72b-instruct.txt",
        help="Path to summary text file that lists per-image utilities.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/wenjie_jacky_mo/emergent-values/utility_analysis/outputs/wiki_subset_72b",
        help="Directory to write CSV and plots.",
    )
    parser.add_argument(
        "--weight_epsilon",
        type=float,
        default=1e-3,
        help="Small value added to variance for stable precision weights.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Parse and aggregate
    image_utils = parse_summary_file(args.summary)
    aggregates, style_to_means = compute_style_aggregates(image_utils, weight_epsilon=args.weight_epsilon)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "style_utility_summary.csv")
    save_csv(aggregates, csv_path)

    # Plots
    plot_bar_with_ci(aggregates, args.out_dir, use_weighted=False)
    plot_bar_with_ci(aggregates, args.out_dir, use_weighted=True)
    plot_boxplot(style_to_means, aggregates, args.out_dir)

    # Print top/bottom few styles for quick glance
    print("Saved:")
    print(f"- CSV: {csv_path}")
    print(f"- Plots: {os.path.join(args.out_dir, 'style_utility_bar.png')}")
    print(f"         {os.path.join(args.out_dir, 'style_utility_bar_weighted.png')}")
    print(f"         {os.path.join(args.out_dir, 'style_utility_box.png')}")
    print()
    print("Top 5 styles by mean utility:")
    for ag in aggregates[:5]:
        print(f"  {ag.style}: mean={ag.mean_of_means:.3f} (n={ag.num_images})")
    print("Bottom 5 styles by mean utility:")
    for ag in aggregates[-5:]:
        print(f"  {ag.style}: mean={ag.mean_of_means:.3f} (n={ag.num_images})")


if __name__ == "__main__":
    main() 