#!/usr/bin/env python3
"""
Generate three synonyms for each item, ensuring no word overlap.
"""

import pandas as pd
import json
import time
import os
import re
import argparse
from anthropic import Anthropic

def get_words(text):
    """Extract lowercase words from text."""
    return set(re.findall(r'\b[a-z]+\b', text.lower()))

def has_word_overlap(text1, text2):
    """Check if two texts share any words."""
    words1 = get_words(text1)
    words2 = get_words(text2)
    return bool(words1 & words2)

def validate_synonyms(original, syn1, syn2, syn3):
    """Check that no synonyms share words with original or each other."""
    original_words = get_words(original)
    syn1_words = get_words(syn1)
    syn2_words = get_words(syn2)
    syn3_words = get_words(syn3)
    
    issues = []
    
    # Check overlap with original
    if syn1_words & original_words:
        issues.append(f"Synonym 1 shares words with original: {syn1_words & original_words}")
    if syn2_words & original_words:
        issues.append(f"Synonym 2 shares words with original: {syn2_words & original_words}")
    if syn3_words & original_words:
        issues.append(f"Synonym 3 shares words with original: {syn3_words & original_words}")
    
    # Check overlap between synonyms
    if syn1_words & syn2_words:
        issues.append(f"Synonym 1 and 2 share words: {syn1_words & syn2_words}")
    if syn1_words & syn3_words:
        issues.append(f"Synonym 1 and 3 share words: {syn1_words & syn3_words}")
    if syn2_words & syn3_words:
        issues.append(f"Synonym 2 and 3 share words: {syn2_words & syn3_words}")
    
    return issues

def generate_synonyms(client, item, max_retries=3):
    """Generate 3 non-overlapping synonyms for an item."""
    
    prompt = f"""Generate exactly 3 synonyms or alternative phrases for the concept: "{item}"

CRITICAL RULES:
1. Each synonym must convey the same or very similar meaning as "{item}"
2. NO synonym can share ANY word with the original term "{item}"
3. NO synonym can share ANY word with any other synonym
4. Keep synonyms concise (1-4 words max)
5. Use common, easily understood language

For example:
- Original: "Emotional control" 
- Good synonyms: "Affect regulation", "Feeling management", "Mood moderation"
- Bad: "Emotional regulation" (shares "emotional"), "Affect management, Feeling management" (both have "management")

Respond in JSON format only:
{{
    "synonym_1": "first alternative",
    "synonym_2": "second alternative", 
    "synonym_3": "third alternative"
}}"""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                syn1 = result.get('synonym_1', '').strip()
                syn2 = result.get('synonym_2', '').strip()
                syn3 = result.get('synonym_3', '').strip()
                
                # Validate
                issues = validate_synonyms(item, syn1, syn2, syn3)
                
                if not issues:
                    return syn1, syn2, syn3, None
                else:
                    if attempt < max_retries - 1:
                        # Retry with more specific instructions
                        prompt = f"""Generate exactly 3 synonyms for: "{item}"

PREVIOUS ATTEMPT FAILED because: {'; '.join(issues)}

STRICT RULES:
1. NO synonym can share ANY word with "{item}" (not even small words like "of", "and", etc.)
2. NO synonym can share ANY word with any other synonym
3. Each must mean the same thing as "{item}"
4. Keep them short (1-4 words)

Respond ONLY with JSON:
{{
    "synonym_1": "first alternative",
    "synonym_2": "second alternative", 
    "synonym_3": "third alternative"
}}"""
                        continue
                    else:
                        return syn1, syn2, syn3, issues
            else:
                if attempt < max_retries - 1:
                    continue
                return None, None, None, ["Failed to parse JSON response"]
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return None, None, None, [str(e)]
    
    return None, None, None, ["Max retries exceeded"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', required=True, help='Anthropic API key')
    parser.add_argument('--input', default='data/facets - 10. new items (104).csv')
    parser.add_argument('--output', default='data/facets - 11. new items with synonyms (104).csv')
    args = parser.parse_args()
    
    # Load input
    df = pd.read_csv(args.input)
    items = df['Items'].tolist()
    print(f"Loaded {len(items)} items")
    
    # Initialize client
    client = Anthropic(api_key=args.api_key)
    
    # Generate synonyms for each item
    results = []
    issues_log = []
    
    for i, item in enumerate(items):
        print(f"[{i+1}/{len(items)}] {item}...", end=" ", flush=True)
        
        syn1, syn2, syn3, issues = generate_synonyms(client, item)
        
        if issues:
            print(f"ISSUES: {issues}")
            issues_log.append({'item': item, 'issues': issues})
        else:
            print(f"OK: {syn1} | {syn2} | {syn3}")
        
        results.append({
            'Items': item,
            'Synonym 1': syn1 or '',
            'Synonym 2': syn2 or '',
            'Synonym 3': syn3 or ''
        })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save output
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    
    # Report issues
    if issues_log:
        print(f"\n{len(issues_log)} items had validation issues:")
        for entry in issues_log:
            print(f"  - {entry['item']}: {entry['issues']}")

if __name__ == '__main__':
    main()
