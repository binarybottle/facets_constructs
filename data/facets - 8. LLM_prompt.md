# Prompt for LLM-Based Construct Assignment (New Session)

Use this prompt to recreate the LLM assignment pipeline in a new coding session.

---

## Full Prompt

```
I have psychological/educational assessment constructs that need to be assigned to standardized categories.

**Input files:**
1. "facets - 7. consolidated (146).csv" - 146 FACETS anchor categories with columns: Subcategory, Source, Category, Brief Description, Description
2. "facets - 6. similarities (876).csv" - 876 candidate items to assign, with columns including: Subcategory, Source, Category, Brief Description, Description, irrelevant (1 = exclude)

**Task:**
Create a Python script that uses an LLM (Claude) to assign each candidate item to one of the 146 FACETS anchors.

**Method:**
1. Filter out items where irrelevant = 1
2. For EACH of the ~803 remaining items, make ONE LLM API call:
   - Show the item's name and description
   - Show the list of 146 FACETS anchors with their brief descriptions
   - Ask: "Which FACETS category does this item best match?"
   - Get response with: assigned_anchor, confidence (high/medium/low), reasoning

3. Group results by assigned anchor and output CSV with columns:
   - Items: comma-delimited list (duplicates as "Name (count)")
   - Sources: comma-delimited as "[Source] ([Category])"
   - Brief Descriptions: "///"-delimited
   - Descriptions: "///"-delimited

**Requirements:**
- One LLM call per item (not batched) for full attention to each assignment
- Save progress incrementally (can resume if interrupted)
- Track confidence levels for quality assurance
- Handle items with no good match (assign to null)
- Output both the grouped CSV and a JSON file with all individual assignments

**Why one-at-a-time:**
- Each item gets full context and attention
- More accurate than batching (no confusion between items)
- Auditable individual decisions
- Cost is minimal (~$3-5 for 800 calls)

Please also provide:
1. A publication-ready method description
2. The prompt that could recreate this in a new session
```

---

## Key Commands

```bash
# Install dependencies
pip install anthropic pandas

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Run assignment (full)
python assign_items_llm.py \
    --anchors "data/facets - 7. consolidated (146).csv" \
    --items "data/facets - 6. similarities (876).csv" \
    --output "data/facets - 10. llm assigned variants (146xN).csv"

# Run test (first 10 items)
python assign_items_llm.py --limit 10
```

---

## Expected Output Files

1. **`assign_items_llm.py`** - Main assignment script
2. **`LLM_ASSIGNMENT_METHOD.md`** - Publication-ready method description
3. **`facets - 10. llm assigned variants (146xN).csv`** - Grouped output (146 rows)
4. **`facets - 10. llm assigned variants (146xN)_assignments.json`** - Individual assignments for auditing

---

## Method Summary (for publication)

> Items were assigned to FACETS categories using large language model (LLM) judgment (Claude Sonnet, Anthropic, 2024). For each of the N candidate constructs, we presented the LLM with the item's name and description alongside the complete list of 146 FACETS anchors. The LLM was instructed to identify the best-matching FACETS category based on semantic similarity, or indicate if no appropriate match existed. Each assignment included a confidence rating (high = same construct, medium = closely related, low = weak match) and brief reasoning. This approach avoids arbitrary similarity thresholds while providing auditable, reproducible assignments. Of the N items evaluated, X (Y%) were assigned with high confidence, Z (W%) with medium confidence, and V items had no appropriate FACETS match.

---

## Comparison to Automated Methods

| Method | Arbitrary Parameters | Human-like Judgment | Auditable |
|--------|---------------------|---------------------|-----------|
| SBERT + threshold | Yes (threshold value) | No | No |
| SBERT + clustering | Yes (cluster count) | No | No |
| Louvain community | No | No | Limited |
| **LLM assignment** | **No** | **Yes** | **Yes** |
