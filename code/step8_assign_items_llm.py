#!/usr/bin/env python3
"""
Assign items to FACETS anchors using LLM-based semantic matching.

Method:
1. Load 146 consolidated FACETS anchors
2. Load similarity items (excluding irrelevant)
3. For each item, ask LLM: "Which FACETS anchor does this belong to?"
4. Group by assignment to create expanded variants file

This approach uses LLM judgment for each assignment, avoiding reliance on
potentially flawed automated similarity measures. Each assignment is independent
and auditable.

Requires: ANTHROPIC_API_KEY environment variable
"""

import pandas as pd
import json
import argparse
import os
import time
import re
from collections import defaultdict
from typing import Optional, Dict, List

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def normalize_name(text: str) -> str:
    """Normalize item name for matching."""
    if pd.isna(text) or not text:
        return ""
    normalized = str(text).lower().strip()
    normalized = normalized.replace('-', ' ').replace('_', ' ')
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def create_display_source(row) -> str:
    """Create display source in format '[Source] ([Category name])'."""
    source = row.get('Source', '')
    category = row.get('Category', '')
    
    if pd.isna(source) or not source:
        return ''
    
    if pd.notna(category) and category:
        return f"{source} ({category})"
    return source


def create_anchor_list(anchors_df: pd.DataFrame) -> str:
    """Create a formatted list of anchors for the prompt."""
    lines = []
    for _, row in anchors_df.iterrows():
        name = row['Subcategory']
        brief = row.get('Brief Description', '')
        if pd.notna(brief) and brief:
            lines.append(f"- {name}: {brief[:100]}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def assign_item_to_anchor(
    client,
    item_name: str,
    item_description: str,
    item_brief: str,
    anchor_list: str,
    anchor_names: List[str],
    model: str = "claude-sonnet-4-20250514"
) -> Dict:
    """
    Ask LLM to assign a single item to a FACETS anchor.
    
    Returns dict with:
    - assigned_anchor: str or None
    - confidence: str (high/medium/low)
    - reasoning: str
    """
    # Build item description
    item_info = f"**{item_name}**"
    if item_brief:
        item_info += f"\nBrief: {item_brief}"
    if item_description and item_description != item_brief:
        item_info += f"\nFull description: {item_description[:300]}"
    
    prompt = f"""You are assigning psychological/educational assessment constructs to standardized categories.

## Item to Assign:
{item_info}

## Available FACETS Categories (146 total):
{anchor_list}

## Task:
Which FACETS category does this item best match? The item should measure the same or a very closely related construct.

Respond in JSON format:
{{
  "assigned_anchor": "exact FACETS name" or null if no good match,
  "confidence": "high" | "medium" | "low",
  "reasoning": "brief explanation (1 sentence)"
}}

Rules:
- Use the EXACT anchor name from the list above
- Only assign if there's a genuine semantic match (same construct)
- Return null for assigned_anchor if no good match exists
- "high" confidence = clearly the same construct
- "medium" confidence = closely related but not identical
- "low" confidence = weak match, consider null instead"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Validate anchor name
            if result.get('assigned_anchor'):
                # Try to match to actual anchor names
                assigned = result['assigned_anchor']
                norm_assigned = normalize_name(assigned)
                
                # Find best match
                matched = None
                for anchor in anchor_names:
                    if normalize_name(anchor) == norm_assigned:
                        matched = anchor
                        break
                
                if not matched:
                    # Try partial match
                    for anchor in anchor_names:
                        if norm_assigned in normalize_name(anchor) or normalize_name(anchor) in norm_assigned:
                            matched = anchor
                            break
                
                result['assigned_anchor'] = matched
            
            return result
        
        return {"assigned_anchor": None, "confidence": "low", "reasoning": "Could not parse response"}
    
    except Exception as e:
        return {"assigned_anchor": None, "confidence": "low", "reasoning": f"Error: {str(e)}"}


def assign_all_items(
    anchors_file: str,
    similarities_file: str,
    output_file: str,
    api_key: str = None,
    model: str = "claude-sonnet-4-20250514",
    limit: int = None,
    save_interval: int = 50
):
    """
    Assign all items to FACETS anchors using LLM.
    """
    if not HAS_ANTHROPIC:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return
    
    api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print("=" * 70)
    print("LLM-BASED ITEM ASSIGNMENT")
    print("=" * 70)
    
    # Load anchors
    print(f"\nLoading anchors from: {anchors_file}")
    anchors_df = pd.read_csv(anchors_file)
    anchor_names = anchors_df['Subcategory'].tolist()
    anchor_list = create_anchor_list(anchors_df)
    print(f"  Loaded {len(anchor_names)} anchors")
    
    # Load similarities
    print(f"\nLoading items from: {similarities_file}")
    items_df = pd.read_csv(similarities_file)
    
    # Filter out irrelevant
    items_df = items_df[items_df['irrelevant'] != 1].reset_index(drop=True)
    print(f"  Loaded {len(items_df)} items (after filtering irrelevant)")
    
    if limit:
        items_df = items_df.head(limit)
        print(f"  Limited to {limit} items for testing")
    
    # Check for existing progress
    assignments_file = output_file.replace('.csv', '_assignments.json')
    assignments = {}
    if os.path.exists(assignments_file):
        with open(assignments_file, 'r') as f:
            assignments = json.load(f)
        print(f"  Loaded {len(assignments)} existing assignments")
    
    # Assign each item
    print(f"\nAssigning items to anchors...")
    print(f"  Model: {model}")
    print(f"  Save interval: every {save_interval} items")
    print("-" * 70)
    
    total = len(items_df)
    assigned_count = 0
    skipped_count = 0
    
    for i, row in items_df.iterrows():
        item_key = f"{row['Subcategory']}|{row['Source']}"
        
        # Skip if already assigned
        if item_key in assignments:
            skipped_count += 1
            continue
        
        item_name = row['Subcategory']
        item_brief = row.get('Brief Description', '') if pd.notna(row.get('Brief Description', '')) else ''
        item_desc = row.get('Description', '') if pd.notna(row.get('Description', '')) else ''
        
        result = assign_item_to_anchor(
            client, item_name, item_desc, item_brief,
            anchor_list, anchor_names, model
        )
        
        assignments[item_key] = {
            'item': item_name,
            'source': row['Source'],
            'category': row.get('Category', ''),
            'brief_description': item_brief,
            'description': item_desc,
            **result
        }
        
        assigned_count += 1
        
        # Progress
        status = result.get('assigned_anchor', 'none')[:30] if result.get('assigned_anchor') else 'NONE'
        conf = result.get('confidence', '?')
        print(f"  [{i+1}/{total}] {item_name[:35]:35} -> {status:30} ({conf})")
        
        # Save periodically
        if assigned_count % save_interval == 0:
            with open(assignments_file, 'w') as f:
                json.dump(assignments, f, indent=2)
            print(f"  ... saved {len(assignments)} assignments")
        
        # Rate limiting
        time.sleep(0.3)
    
    # Final save
    with open(assignments_file, 'w') as f:
        json.dump(assignments, f, indent=2)
    print(f"\nAssignments saved to: {assignments_file}")
    
    # Generate grouped output
    print("\nGenerating grouped output...")
    generate_grouped_output(anchors_df, assignments, output_file)
    
    # Statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total items: {total}")
    print(f"New assignments: {assigned_count}")
    print(f"Skipped (existing): {skipped_count}")
    
    assigned_items = [a for a in assignments.values() if a.get('assigned_anchor')]
    unassigned = [a for a in assignments.values() if not a.get('assigned_anchor')]
    print(f"Assigned to anchor: {len(assigned_items)}")
    print(f"No match (none): {len(unassigned)}")
    
    confidence_counts = defaultdict(int)
    for a in assigned_items:
        confidence_counts[a.get('confidence', 'unknown')] += 1
    print(f"Confidence: high={confidence_counts['high']}, medium={confidence_counts['medium']}, low={confidence_counts['low']}")


def generate_grouped_output(anchors_df: pd.DataFrame, assignments: Dict, output_file: str):
    """Generate the grouped variants output file."""
    
    # Group assignments by anchor
    anchor_items = defaultdict(list)
    for key, assignment in assignments.items():
        anchor = assignment.get('assigned_anchor')
        if anchor:
            anchor_items[anchor].append(assignment)
    
    # Build output rows
    output_rows = []
    
    for _, anchor_row in anchors_df.iterrows():
        anchor_name = anchor_row['Subcategory']
        anchor_source = create_display_source(anchor_row)
        anchor_brief = anchor_row.get('Brief Description', '') if pd.notna(anchor_row.get('Brief Description', '')) else ''
        anchor_desc = anchor_row.get('Description', '') if pd.notna(anchor_row.get('Description', '')) else ''
        
        # Start with the anchor itself
        items_dict = defaultdict(lambda: {'count': 0, 'sources': [], 'briefs': [], 'descs': []})
        
        norm_anchor = normalize_name(anchor_name)
        items_dict[norm_anchor] = {
            'original_name': anchor_name,
            'count': 1,
            'sources': [anchor_source],
            'briefs': [anchor_brief],
            'descs': [anchor_desc]
        }
        
        # Add assigned items
        for assignment in anchor_items.get(anchor_name, []):
            item_name = assignment['item']
            norm_name = normalize_name(item_name)
            
            source = assignment.get('source', '')
            category = assignment.get('category', '')
            if category:
                display_source = f"{source} ({category})"
            else:
                display_source = source
            
            brief = assignment.get('brief_description', '')
            desc = assignment.get('description', '')
            
            if norm_name not in items_dict:
                items_dict[norm_name] = {
                    'original_name': item_name,
                    'count': 0,
                    'sources': [],
                    'briefs': [],
                    'descs': []
                }
            
            # Deduplicate by source
            if display_source not in items_dict[norm_name]['sources']:
                items_dict[norm_name]['count'] += 1
                items_dict[norm_name]['sources'].append(display_source)
                items_dict[norm_name]['briefs'].append(brief)
                items_dict[norm_name]['descs'].append(desc)
        
        # Format output
        items = []
        all_sources = []
        all_briefs = []
        all_descs = []
        
        for norm_name, data in items_dict.items():
            name = data['original_name']
            count = data['count']
            
            if count > 1:
                items.append(f"{name} ({count})")
            else:
                items.append(name)
            
            all_sources.extend(data['sources'])
            all_briefs.extend(data['briefs'])
            all_descs.extend(data['descs'])
        
        output_rows.append({
            'Items': ', '.join(items),
            'Sources': ', '.join(all_sources),
            'Brief Descriptions': '///'.join(all_briefs),
            'Descriptions': '///'.join(all_descs)
        })
    
    # Save
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_file, index=False)
    print(f"Output saved to: {output_file}")
    print(f"  Total rows: {len(output_df)}")
    
    # Stats
    items_per_anchor = [len(row['Items'].split(', ')) for _, row in output_df.iterrows()]
    print(f"  Items per anchor: min={min(items_per_anchor)}, max={max(items_per_anchor)}, mean={sum(items_per_anchor)/len(items_per_anchor):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Assign items to FACETS anchors using LLM')
    parser.add_argument('--anchors', '-a', type=str,
                       default='data/facets - 7. consolidated (146).csv',
                       help='CSV file with FACETS anchors')
    parser.add_argument('--items', '-i', type=str,
                       default='data/facets - 6. similarities (876).csv',
                       help='CSV file with items to assign')
    parser.add_argument('--output', '-o', type=str,
                       default='data/facets - 10. llm assigned variants (146xN).csv',
                       help='Output CSV file')
    parser.add_argument('--api-key', type=str, default=None,
                       help='Anthropic API key')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-20250514',
                       help='Model to use')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of items (for testing)')
    parser.add_argument('--save-interval', type=int, default=50,
                       help='Save progress every N items')
    
    args = parser.parse_args()
    
    assign_all_items(
        anchors_file=args.anchors,
        similarities_file=args.items,
        output_file=args.output,
        api_key=args.api_key,
        model=args.model,
        limit=args.limit,
        save_interval=args.save_interval
    )
