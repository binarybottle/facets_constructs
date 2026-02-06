# LLM-Based Construct Assignment Method

## Overview

This document describes a method for assigning psychological and educational assessment constructs to standardized FACETS categories using large language model (LLM) judgment. Each construct is independently evaluated for semantic similarity to 146 pre-defined FACETS anchors, producing auditable, reproducible assignments.

## Rationale

### Limitations of Automated Similarity Methods

Prior approaches using Sentence-BERT embeddings and cosine similarity showed several limitations:

1. **Misattribution**: 72 of 876 items (8.2%) were flagged as incorrectly assigned by human review
2. **Surface-level matching**: SBERT sometimes matched based on word overlap rather than conceptual similarity (e.g., "Aggression & Conflict" → "Assertiveness")
3. **Threshold dependency**: Clustering methods require arbitrary cutoffs for similarity scores or cluster counts

### Advantages of LLM-Based Assignment

1. **Semantic understanding**: LLMs can distinguish conceptually similar constructs from superficially similar ones
2. **Consistent methodology**: All assignments use identical evaluation criteria
3. **No arbitrary parameters**: No thresholds, cluster counts, or cutoffs
4. **Auditable**: Each assignment includes confidence rating and reasoning
5. **Handles edge cases**: Items with no good match are explicitly identified

## Method

### Input Data

1. **FACETS Anchors** (N=146): Consolidated list of standardized psychological/educational constructs with names and descriptions
2. **Candidate Items** (N=803): Constructs from various sources (clinical scales, frameworks, AI-generated) to be assigned to anchors

### Assignment Procedure

For each candidate item:

1. **Construct prompt** containing:
   - The item's name, brief description, and full description
   - Complete list of 146 FACETS anchors with brief descriptions

2. **Query LLM** with instruction:
   > "Which FACETS category does this item best match? The item should measure the same or a very closely related construct."

3. **Parse response** containing:
   - `assigned_anchor`: Exact name of matched FACETS category, or null if no good match
   - `confidence`: "high" (same construct), "medium" (closely related), or "low" (weak match)
   - `reasoning`: One-sentence explanation

4. **Validate anchor name** against the list of valid FACETS names

### Confidence Criteria

| Level | Definition | Action |
|-------|------------|--------|
| High | Item measures the same construct as the anchor | Include in group |
| Medium | Item measures a closely related construct | Include in group |
| Low | Weak or uncertain match | Include with caution |
| Null | No appropriate anchor exists | Exclude from groupings |

### Output Format

Items are grouped by their assigned anchor into a CSV with columns:

- **Items**: Comma-delimited list of construct names (duplicates shown with count, e.g., "Self-Awareness (3)")
- **Sources**: Comma-delimited list as "[Source] ([Category])"
- **Brief Descriptions**: "///"-delimited descriptions
- **Descriptions**: "///"-delimited full descriptions

## Implementation

### Requirements

- Python 3.8+
- anthropic (Anthropic API client)
- pandas (data manipulation)
- ANTHROPIC_API_KEY environment variable

### Usage

```bash
export ANTHROPIC_API_KEY="your-api-key"

python assign_items_llm.py \
    --anchors "data/facets - 7. consolidated (146).csv" \
    --items "data/facets - 6. similarities (876).csv" \
    --output "data/facets - 10. llm assigned variants (146xN).csv" \
    --model "claude-sonnet-4-20250514"
```

### Progress Tracking

- Assignments are saved incrementally (default: every 50 items)
- If interrupted, the script resumes from the last saved checkpoint
- Individual assignments stored in `*_assignments.json` for auditing

### Cost Estimate

- ~803 API calls (one per item)
- ~1000 tokens per call (prompt + response)
- Using Claude Sonnet: approximately $3-5 total

## Quality Assurance

### Validation

1. **Anchor name validation**: Ensures assigned names match exactly to valid FACETS categories
2. **Confidence tracking**: Low-confidence assignments flagged for review
3. **Null handling**: Items with no good match explicitly identified

### Comparison to Automated Methods

The LLM-based approach can be validated against:
- Human expert assignments (gold standard)
- SBERT-based assignments (for items not flagged as misattributed)
- Coarse/granular labels (for subset with human annotations)

## Reproducibility

### Factors Affecting Reproducibility

1. **Model version**: Results may vary across model versions
2. **Temperature**: Set to 0 (deterministic) by default
3. **Prompt wording**: Standardized prompt ensures consistency

### Recommended Citation

When using this method, cite:
- The specific LLM model and version used
- The date of assignment
- This methodology document

## Results Summary

For the FACETS construct assignment task:

- **Input**: 803 candidate items, 146 FACETS anchors
- **Assigned**: [N] items to anchors
- **Unassigned**: [M] items (no good match)
- **Confidence distribution**: High=[X]%, Medium=[Y]%, Low=[Z]%

## References

Anthropic. (2024). Claude models documentation. https://docs.anthropic.com/

Brown, T., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33.
