#!/usr/bin/env python3
"""
Convert JSONL sample files to the final JSON format used for experiments.

Usage:
    python convert_jsonl_to_final_json.py input.jsonl output.json

Format conversion:
    JSONL: {"path": "/path/to/image.jpg", "class": "n01443537", ...}
    JSON: {"Images": [{"images": ["/path/to/image.jpg"]}, ...]}
"""

import json
import sys
from pathlib import Path
from typing import List, Dict


def convert_jsonl_to_final_json(jsonl_path: str, output_json_path: str):
    """
    Convert a JSONL file to the final JSON format.

    Args:
        jsonl_path: Path to input JSONL file
        output_json_path: Path to output JSON file
    """
    jsonl_path = Path(jsonl_path)
    output_json_path = Path(output_json_path)

    if not jsonl_path.exists():
        print(f"Error: Input file {jsonl_path} does not exist")
        sys.exit(1)

    # Read all entries from JSONL
    images_list = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            # Extract the path (should be in 'path' field)
            if 'path' in entry:
                images_list.append({
                    "images": [entry['path']]
                })
            else:
                print(f"Warning: Entry missing 'path' field: {entry}")

    # Create final JSON structure
    final_json = {
        "Images": images_list
    }

    # Write to output file
    with open(output_json_path, 'w') as f:
        json.dump(final_json, f, indent=2)

    print(f"Converted {len(images_list)} images from {jsonl_path} to {output_json_path}")


def convert_all_in_directory(directory: str = None):
    """
    Convert all JSONL files in a directory to final JSON format.

    Args:
        directory: Directory to process (default: current directory)
    """
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)

    # Find all JSONL files
    jsonl_files = list(directory.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {directory}")
        return

    print(f"Found {len(jsonl_files)} JSONL files to convert")

    for jsonl_file in jsonl_files:
        # Create output filename: replace .jsonl with _final.json
        output_name = jsonl_file.stem + "_final.json"
        output_path = jsonl_file.parent / output_name

        print(f"\nConverting: {jsonl_file.name} -> {output_name}")
        convert_jsonl_to_final_json(str(jsonl_file), str(output_path))

    print(f"\nConversion complete!")


def main():
    if len(sys.argv) == 1:
        # No arguments: convert all JSONL files in current directory
        print("No arguments provided. Converting all JSONL files in current directory...")
        convert_all_in_directory()

    elif len(sys.argv) == 2:
        if sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            print("\nUsage options:")
            print("  1. Convert single file:")
            print("     python convert_jsonl_to_final_json.py input.jsonl output.json")
            print("\n  2. Convert all JSONL files in a directory:")
            print("     python convert_jsonl_to_final_json.py /path/to/directory")
            print("\n  3. Convert all JSONL files in current directory:")
            print("     python convert_jsonl_to_final_json.py")
            sys.exit(0)
        else:
            # Single argument: treat as directory
            convert_all_in_directory(sys.argv[1])

    elif len(sys.argv) == 3:
        # Two arguments: input and output files
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_jsonl_to_final_json(input_file, output_file)

    else:
        print("Error: Too many arguments")
        print("Usage: python convert_jsonl_to_final_json.py [input.jsonl] [output.json]")
        print("       python convert_jsonl_to_final_json.py [directory]")
        print("       python convert_jsonl_to_final_json.py (converts all in current dir)")
        sys.exit(1)


if __name__ == "__main__":
    main()
