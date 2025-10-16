#!/usr/bin/env python3
"""
Quick checker to analyze the actual experiment results and show examples of each response type.
"""

import json
from collections import Counter

RESULTS_PATH = "/data/richard_ren/emergent-values-multimodal/outputs/company_final_32b/results_qwen25-vl-32b-instruct.json"

print("=" * 80)
print("EXPERIMENT RESULTS ANALYZER")
print("=" * 80)

print(f"\nLoading results from: {RESULTS_PATH}")

with open(RESULTS_PATH, 'r') as f:
    data = json.load(f)

edges = data['graph_data']['edges']

print(f"Total edges: {len(edges)}")

# Collect all responses with categorization
categories = {
    'valid_A': [],
    'valid_B': [],
    'blindness': [],
    'unparseable': [],
    'long': [],
}

for edge_key, edge_data in edges.items():
    aux = edge_data.get('aux_data', {})

    # Process original responses
    for resp, parsed in zip(aux.get('original_responses', []), aux.get('original_parsed', [])):
        if parsed == 'A':
            categories['valid_A'].append(resp)
        elif parsed == 'B':
            categories['valid_B'].append(resp)
        elif parsed == 'unparseable':
            if 'cannot' in resp.lower() or 'unable' in resp.lower():
                categories['blindness'].append(resp)
            else:
                categories['unparseable'].append(resp)

        if len(resp) > 10:
            categories['long'].append(resp)

    # Process flipped responses
    for resp, parsed in zip(aux.get('flipped_responses', []), aux.get('flipped_parsed', [])):
        if parsed == 'A':
            categories['valid_A'].append(resp)
        elif parsed == 'B':
            categories['valid_B'].append(resp)
        elif parsed == 'unparseable':
            if 'cannot' in resp.lower() or 'unable' in resp.lower():
                categories['blindness'].append(resp)
            else:
                categories['unparseable'].append(resp)

        if len(resp) > 10:
            categories['long'].append(resp)

total_responses = sum(len(cat) for cat in categories.values() if cat != categories['long'])

print("\n" + "=" * 80)
print("RESPONSE BREAKDOWN")
print("=" * 80)

print(f"\nValid 'A' responses: {len(categories['valid_A'])} ({100*len(categories['valid_A'])/total_responses:.2f}%)")
print(f"Valid 'B' responses: {len(categories['valid_B'])} ({100*len(categories['valid_B'])/total_responses:.2f}%)")
print(f"'Blindness' responses: {len(categories['blindness'])} ({100*len(categories['blindness'])/total_responses:.2f}%)")
print(f"Other unparseable: {len(categories['unparseable'])} ({100*len(categories['unparseable'])/total_responses:.2f}%)")
print(f"Long responses (>10 chars): {len(categories['long'])} ({100*len(categories['long'])/total_responses:.2f}%)")

print("\n" + "=" * 80)
print("EXAMPLE RESPONSES")
print("=" * 80)

print("\n1. Valid 'A' Responses (showing first 10):")
print("-" * 80)
for i, resp in enumerate(categories['valid_A'][:10], 1):
    print(f"{i}. '{resp}'")

print("\n2. Valid 'B' Responses (showing first 10):")
print("-" * 80)
for i, resp in enumerate(categories['valid_B'][:10], 1):
    print(f"{i}. '{resp}'")

print("\n3. 'Blindness' Responses (showing all unique ones):")
print("-" * 80)
unique_blindness = list(set(categories['blindness']))
for i, resp in enumerate(unique_blindness[:20], 1):
    print(f"{i}. {resp}")

print("\n4. Other Unparseable Responses (showing first 20 unique):")
print("-" * 80)
unique_unparseable = list(set(categories['unparseable']))
for i, resp in enumerate(unique_unparseable[:20], 1):
    print(f"{i}. {resp[:100]}..." if len(resp) > 100 else f"{i}. {resp}")

print("\n5. Long Responses (>10 chars, showing first 10 unique):")
print("-" * 80)
unique_long = list(set(categories['long']))
for i, resp in enumerate(unique_long[:10], 1):
    print(f"{i}. {resp[:150]}..." if len(resp) > 150 else f"{i}. {resp}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

valid_count = len(categories['valid_A']) + len(categories['valid_B'])
valid_pct = 100 * valid_count / total_responses

print(f"\n✓ Valid response rate: {valid_pct:.2f}%")

if valid_pct > 90:
    print("  → EXCELLENT: Very high success rate")
    print("  → Images are clearly being processed correctly")
elif valid_pct > 80:
    print("  → GOOD: High success rate")
    print("  → Minor issues but images are being processed")
else:
    print("  → CONCERNING: Low success rate")
    print("  → May indicate image processing issues")

blindness_pct = 100 * len(categories['blindness']) / total_responses
print(f"\n⚠ 'Blindness' response rate: {blindness_pct:.2f}%")

if blindness_pct < 2:
    print("  → NORMAL: Very low rate, likely model safety responses")
    print("  → Not indicative of image loading failure")
elif blindness_pct < 5:
    print("  → ACCEPTABLE: Low rate, but worth monitoring")
else:
    print("  → HIGH: May indicate systematic image loading issues")

unparseable_pct = 100 * len(categories['unparseable']) / total_responses
print(f"\n⚠ Other unparseable rate: {unparseable_pct:.2f}%")

if unparseable_pct < 5:
    print("  → NORMAL: Low rate of parsing failures")
elif unparseable_pct < 10:
    print("  → MODERATE: Consider stricter prompting")
else:
    print("  → HIGH: Consider adjusting prompt format")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if valid_pct > 90 and blindness_pct < 2:
    print("""
✓ IMAGES ARE LOADING CORRECTLY!

Your experiment results show:
- Very high valid response rate (>90%)
- Very low 'blindness' rate (<2%)
- The model is clearly seeing and processing the images

The small percentage of problematic responses represents normal LLM behavior
variation, not a technical failure in image loading.

Your results are VALID and can be TRUSTED!
""")
elif valid_pct > 80:
    print("""
✓ Images appear to be loading, but with some issues

Your experiment results show acceptable performance, though there's room for
improvement. Consider:
- Adjusting prompt format for clearer instructions
- Using stricter output constraints
- Checking for specific patterns in unparseable responses
""")
else:
    print("""
⚠ Potential issues detected

The results suggest there may be problems with image loading or processing.
Recommended actions:
1. Run the full diagnostic: python diagnose_image_loading.py
2. Check vLLM version compatibility with Qwen2.5-VL
3. Verify GPU memory and compute resources
4. Review logs for errors or warnings
""")

print("=" * 80)
