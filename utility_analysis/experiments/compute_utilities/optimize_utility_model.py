#!/usr/bin/env python3

import asyncio
import json
import os
import time
import argparse
import sys
sys.path.append("../../")
from compute_utilities.compute_utilities import compute_utilities
from compute_utilities.templates import comparison_prompt_template_images
from compute_utilities.utils import flatten_hierarchical_options

async def optimize_utility_model(args):
    """
    Compute utilities for all options in a given options file and save them in a structured directory.
    """
    start_time = time.time()

    # Load options
    with open(args.options_path, 'r') as f:
        options_data = json.load(f)

    # If multimodal, coerce options into descriptions that are just placeholders (we send images via messages)
    comparison_template = None
    if args.multimodal:
        comparison_template = comparison_prompt_template_images
        # For multimodal, each option item is expected to be a dict like {"images": [path]} or list[str]
        def to_desc(item):
            if isinstance(item, dict) and 'images' in item and len(item['images']) > 0:
                return item['images'][0]
            elif isinstance(item, str):
                return item
            else:
                return json.dumps(item)
        if isinstance(options_data, dict):
            flat = flatten_hierarchical_options(options_data)
            options_data = [to_desc(x) for x in flat]
        elif isinstance(options_data, list):
            options_data = [to_desc(x) for x in options_data]

    # Compute utilities
    print(f"\nComputing utilities for {args.options_path}...")
    utility_results = await compute_utilities(
        options_list=options_data,
        model_key=args.model_key,
        create_agent_config_path=args.create_agent_config_path,
        create_agent_config_key=args.create_agent_config_key,
        compute_utilities_config_path=args.compute_utilities_config_path,
        compute_utilities_config_key=args.compute_utilities_config_key,
        save_dir=args.save_dir,
        save_suffix=args.save_suffix,
        with_reasoning=args.with_reasoning,
        comparison_prompt_template=comparison_template
    )

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    return utility_results

async def main():
    parser = argparse.ArgumentParser(description="Compute and save utilities for a given options file.")
    parser.add_argument("--model_key", default="gpt-4o", help="Model key to use")
    parser.add_argument("--save_dir", default="../shared_utilities", help="Base directory to save results")
    parser.add_argument("--save_suffix", default=None, help="Custom suffix for saved files")
    parser.add_argument("--options_path", default="../shared_options/options_hierarchical.json", help="Path to options file")
    parser.add_argument("--with_reasoning", action="store_true", help="Whether to use reasoning in prompts")
    parser.add_argument("--compute_utilities_config_path", default="../compute_utilities.yaml", help="Path to compute_utilities.yaml")
    parser.add_argument("--compute_utilities_config_key", default="default", help="Key to use in compute_utilities.yaml")
    parser.add_argument("--create_agent_config_path", default="../create_agent.yaml", help="Path to create_agent.yaml")
    parser.add_argument("--create_agent_config_key", default=None, help="Key to use in create_agent.yaml (if None, uses 'default_with_reasoning' if with_reasoning=True, else 'default')")
    parser.add_argument("--multimodal", action="store_true", help="Use image-based utility prompts and options")
    args = parser.parse_args()

    await optimize_utility_model(args)

if __name__ == "__main__":
    asyncio.run(main()) 