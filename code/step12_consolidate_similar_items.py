#!/usr/bin/env python3
"""
Consolidate semantically similar items into a smaller set using an LLM.
"""

import pandas as pd
import json
import argparse
from anthropic import Anthropic

def consolidate_items(client, items):
    """Use LLM to identify and consolidate semantically similar items."""
    
    items_list = "\n".join([f"{i+1}. {item}" for i, item in enumerate(items)])
    
    prompt = f"""You are consolidating a list of psychological/educational constructs by merging semantically similar or overlapping items.

## Current Items (104 total):
{items_list}

## Task:
Identify groups of items that are semantically similar, overlapping, or redundant, and consolidate them into a smaller set. For each group:
1. Choose the best representative term (can be one of the originals or a new encompassing term)
2. List which original items it consolidates

Guidelines:
- Items measuring the same underlying construct should be merged (e.g., "Managing emotions" and "Regulating emotions")
- Items that are subsets of broader constructs can be merged (e.g., "Morning routine" into "Habits / routines")
- Keep items that are truly distinct separate
- Aim to reduce the list significantly while preserving important distinctions
- The "Relationships with X" items could potentially be consolidated
- The "Self-X" items may have some overlap but also distinct meanings
- Consider whether similar concepts like "Resilience", "Coping skills", "Frustration tolerance" should be merged

Respond in JSON format:
{{
    "consolidated_items": [
        {{
            "representative_term": "term that best represents the group",
            "original_items": ["item1", "item2", ...],
            "rationale": "brief explanation of why these were merged"
        }},
        ...
    ],
    "standalone_items": ["items that remain unchanged because they're distinct"]
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    
    # Extract JSON
    json_start = text.find('{')
    json_end = text.rfind('}') + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(text[json_start:json_end])
    
    raise ValueError("Could not parse LLM response as JSON")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='Anthropic API key')
    parser.add_argument('--input', default='data/facets - 10. candidate items (104).csv')
    parser.add_argument('--output', default='data/facets - 11. consolidated items.csv')
    parser.add_argument('--output-mapping', default='data/facets - 11. consolidation mapping.csv')
    args = parser.parse_args()
    
    # Load input
    df = pd.read_csv(args.input)
    items = df['Items'].tolist()
    print(f"Loaded {len(items)} items")
    
    # Initialize client
    client = Anthropic(api_key=args.api_key)
    
    print("Asking LLM to consolidate similar items...")
    result = consolidate_items(client, items)
    
    # Build final item list and mapping
    final_items = []
    mapping_rows = []
    
    # Add consolidated items
    for group in result.get('consolidated_items', []):
        rep_term = group['representative_term']
        originals = group['original_items']
        rationale = group.get('rationale', '')
        
        final_items.append(rep_term)
        
        for orig in originals:
            mapping_rows.append({
                'Original Item': orig,
                'Consolidated To': rep_term,
                'Rationale': rationale
            })
    
    # Add standalone items
    for item in result.get('standalone_items', []):
        final_items.append(item)
        mapping_rows.append({
            'Original Item': item,
            'Consolidated To': item,
            'Rationale': 'Standalone - distinct concept'
        })
    
    # Save consolidated items
    out_df = pd.DataFrame({'Items': sorted(final_items)})
    out_df.to_csv(args.output, index=False)
    print(f"\nConsolidated to {len(final_items)} items")
    print(f"Saved to {args.output}")
    
    # Save mapping
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df = mapping_df.sort_values('Consolidated To')
    mapping_df.to_csv(args.output_mapping, index=False)
    print(f"Saved mapping to {args.output_mapping}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Original items: {len(items)}")
    print(f"Consolidated items: {len(final_items)}")
    print(f"Reduction: {len(items) - len(final_items)} items ({100*(len(items)-len(final_items))/len(items):.1f}%)")
    
    print(f"\n=== Consolidations ===")
    for group in result.get('consolidated_items', []):
        print(f"\n{group['representative_term']}:")
        for orig in group['original_items']:
            print(f"  <- {orig}")

if __name__ == '__main__':
    main()
