"""
Analyze redundancies and similarities between FACETS-60 subcategories and other assessment sources.

This script compares subcategories from Clinical scales, LLM-generated sources (Claude Opus 4.5,
GPT 5.2 Thinking, Gemini 3 Pro), and Framework sources against FACETS-60 as the reference.
It generates similarity scores based on subcategory names, brief descriptions, and full descriptions
using a combination of fuzzy string matching, semantic word overlap, and TF-IDF cosine similarity.

Output: A CSV file with additional "Redundant" and "Similarity" columns indicating which rows
have redundant subcategories with FACETS-60 rows.

Example usage:
    python analyze_redundancies.py input.csv output.csv

Author: Generated for assessment subcategory analysis
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')


SEMANTIC_GROUPS = {
    'attention': ['attention', 'focus', 'concentration', 'distractibility', 'sustained', 'selective'],
    'emotion_regulation': ['emotion', 'regulation', 'reactivity', 'emotional', 'feelings', 'affect'],
    'anger': ['anger', 'frustration', 'irritability', 'irritable', 'aggression', 'hostile', 'temper'],
    'anxiety': ['anxiety', 'worry', 'fear', 'nervous', 'anxious', 'worried', 'stress', 'distress'],
    'social': ['social', 'peer', 'friendship', 'relationship', 'interpersonal', 'interaction'],
    'behavior': ['behavior', 'conduct', 'control', 'regulation', 'self-control', 'impulse'],
    'academic': ['academic', 'learning', 'school', 'education', 'study', 'achievement'],
    'communication': ['communication', 'language', 'speech', 'expressive', 'receptive', 'verbal'],
    'motor': ['motor', 'movement', 'physical', 'coordination', 'fine', 'gross'],
    'self': ['self', 'identity', 'esteem', 'worth', 'confidence', 'image', 'concept'],
    'planning': ['planning', 'organization', 'executive', 'task', 'completion', 'management'],
    'memory': ['memory', 'recall', 'remember', 'retention', 'working memory'],
    'thinking': ['thinking', 'cognitive', 'reasoning', 'abstract', 'problem-solving'],
    'empathy': ['empathy', 'compassion', 'kindness', 'perspective', 'caring'],
    'compliance': ['compliance', 'rule', 'following', 'obedience', 'cooperation'],
    'transition': ['transition', 'change', 'flexibility', 'adaptation', 'adjustment'],
    'sleep': ['sleep', 'rest', 'fatigue', 'energy', 'tired', 'drowsy'],
    'eating': ['eating', 'food', 'nutrition', 'diet', 'appetite'],
    'harm': ['harm', 'injury', 'hurt', 'damage', 'violence', 'aggression'],
    'wellbeing': ['wellbeing', 'health', 'wellness', 'functioning'],
}


def clean_text(text: str) -> str:
    """
    Clean and normalize text for comparison.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned and lowercased text with extra whitespace removed
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def fuzzy_ratio(str1: str, str2: str) -> float:
    """
    Compute fuzzy string similarity ratio between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    if not str1 or not str2:
        return 0.0
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def get_semantic_groups(text: str) -> set:
    """
    Identify semantic groups that a text belongs to.
    
    Args:
        text: Input text
        
    Returns:
        Set of semantic group names
    """
    text_lower = clean_text(text)
    groups = set()
    for group_name, keywords in SEMANTIC_GROUPS.items():
        for keyword in keywords:
            if keyword in text_lower:
                groups.add(group_name)
                break
    return groups


def semantic_group_overlap(str1: str, str2: str) -> float:
    """
    Compute semantic group overlap between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Overlap score between 0.0 and 1.0
    """
    groups1 = get_semantic_groups(str1)
    groups2 = get_semantic_groups(str2)
    if not groups1 or not groups2:
        return 0.0
    intersection = groups1 & groups2
    union = groups1 | groups2
    return len(intersection) / len(union) if union else 0.0


def compute_token_overlap(str1: str, str2: str) -> float:
    """
    Compute token-level Jaccard similarity between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Jaccard similarity coefficient between 0.0 and 1.0
    """
    tokens1 = set(clean_text(str1).split())
    tokens2 = set(clean_text(str2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def contains_key_term(str1: str, str2: str) -> float:
    """
    Check if one string contains a key term from another.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        1.0 if containment found, 0.0 otherwise
    """
    clean1 = clean_text(str1)
    clean2 = clean_text(str2)
    words1 = set(w for w in clean1.split() if len(w) > 3)
    words2 = set(w for w in clean2.split() if len(w) > 3)
    
    for w1 in words1:
        for w2 in words2:
            if w1 in w2 or w2 in w1:
                return 1.0
    return 0.0


def compute_tfidf_similarity(texts_a: list, texts_b: list) -> np.ndarray:
    """
    Compute TF-IDF cosine similarity matrix between two sets of texts.
    
    Args:
        texts_a: First list of text strings (rows)
        texts_b: Second list of text strings (columns - reference set)
        
    Returns:
        Similarity matrix of shape (len(texts_a), len(texts_b))
    """
    all_texts = texts_a + texts_b
    cleaned_texts = [clean_text(t) for t in all_texts]
    
    non_empty = [t for t in cleaned_texts if t.strip()]
    if len(non_empty) < 2:
        return np.zeros((len(texts_a), len(texts_b)))
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        stop_words='english'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
        matrix_a = tfidf_matrix[:len(texts_a)]
        matrix_b = tfidf_matrix[len(texts_a):]
        return cosine_similarity(matrix_a, matrix_b)
    except ValueError:
        return np.zeros((len(texts_a), len(texts_b)))


def compute_composite_similarity(
    row_subcategory: str,
    row_brief: str,
    row_desc: str,
    facets_subcategory: str,
    facets_brief: str,
    facets_desc: str,
    tfidf_sim_brief: float,
    tfidf_sim_desc: float
) -> float:
    """
    Compute composite similarity score between a row and a FACETS subcategory.
    
    Uses weighted combination of multiple similarity metrics with emphasis on
    subcategory name matching and semantic overlap.
    
    Args:
        row_subcategory: Subcategory name from the row being compared
        row_brief: Brief description from the row
        row_desc: Full description from the row
        facets_subcategory: FACETS subcategory name
        facets_brief: FACETS brief description
        facets_desc: FACETS full description
        tfidf_sim_brief: Pre-computed TF-IDF similarity for brief descriptions
        tfidf_sim_desc: Pre-computed TF-IDF similarity for full descriptions
        
    Returns:
        Composite similarity score between 0.0 and 1.0
    """
    name_fuzzy = fuzzy_ratio(row_subcategory, facets_subcategory)
    name_token = compute_token_overlap(row_subcategory, facets_subcategory)
    name_contains = contains_key_term(row_subcategory, facets_subcategory)
    name_semantic = semantic_group_overlap(row_subcategory, facets_subcategory)
    
    name_sim = max(
        0.5 * name_fuzzy + 0.3 * name_token + 0.2 * name_semantic,
        0.8 * name_contains + 0.2 * name_semantic,
        name_fuzzy if name_fuzzy > 0.7 else 0
    )
    
    brief_token = compute_token_overlap(row_brief, facets_brief)
    brief_semantic = semantic_group_overlap(row_brief, facets_brief)
    brief_sim = 0.4 * tfidf_sim_brief + 0.3 * brief_token + 0.3 * brief_semantic
    
    desc_semantic = semantic_group_overlap(row_desc, facets_desc)
    desc_sim = 0.6 * tfidf_sim_desc + 0.4 * desc_semantic
    
    composite = 0.45 * name_sim + 0.30 * brief_sim + 0.25 * desc_sim
    
    if name_sim > 0.7:
        composite = max(composite, 0.7 * name_sim + 0.3 * brief_sim)
    
    return min(composite, 1.0)


def analyze_redundancies(input_path: str, output_path: str, threshold: float = 0.35) -> pd.DataFrame:
    """
    Analyze redundancies between FACETS-60 and other assessment sources.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        threshold: Similarity threshold above which items are marked as redundant
        
    Returns:
        DataFrame with added Redundant and Similarity columns
    """
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    
    df.columns = [col.strip() for col in df.columns]
    
    keep_cols = ['Source', 'Category', 'Subcategory', 'Brief Description', 'Description']
    available_cols = [col for col in df.columns if col in keep_cols or col == '']
    available_cols = [col for col in available_cols if col != ''][:5]
    
    if len(available_cols) < 5:
        for col in keep_cols:
            if col not in available_cols:
                df[col] = ''
        available_cols = keep_cols
    
    df = df[available_cols].copy()
    df.columns = keep_cols
    
    df = df.dropna(subset=['Source']).copy()
    df = df[df['Source'].str.strip() != ''].copy()
    df = df.reset_index(drop=True)
    
    for col in ['Subcategory', 'Brief Description', 'Description']:
        df[col] = df[col].fillna('')
    
    facets_mask = df['Source'] == 'FACETS-60'
    facets_df = df[facets_mask].copy()
    other_df = df[~facets_mask].copy()
    
    print(f"FACETS-60 rows: {len(facets_df)}")
    print(f"Other rows: {len(other_df)}")
    print(f"Sources: {df['Source'].unique()}")
    
    facets_subcats = facets_df['Subcategory'].tolist()
    facets_briefs = facets_df['Brief Description'].tolist()
    facets_descs = facets_df['Description'].tolist()
    facets_indices = facets_df.index.tolist()
    
    other_briefs = other_df['Brief Description'].tolist()
    other_descs = other_df['Description'].tolist()
    other_indices = other_df.index.tolist()
    
    print("Computing TF-IDF similarities for brief descriptions...")
    tfidf_sim_brief = compute_tfidf_similarity(other_briefs, facets_briefs)
    
    print("Computing TF-IDF similarities for full descriptions...")
    tfidf_sim_desc = compute_tfidf_similarity(other_descs, facets_descs)
    
    df['Redundant'] = ''
    df['Similarity'] = 0.0
    
    print("Computing composite similarities...")
    for i, idx in enumerate(other_indices):
        row = other_df.loc[idx]
        best_similarity = 0.0
        redundant_with = []
        
        for j, facets_idx in enumerate(facets_indices):
            composite_sim = compute_composite_similarity(
                row['Subcategory'],
                row['Brief Description'],
                row['Description'],
                facets_subcats[j],
                facets_briefs[j],
                facets_descs[j],
                tfidf_sim_brief[i, j],
                tfidf_sim_desc[i, j]
            )
            
            if composite_sim > best_similarity:
                best_similarity = composite_sim
            
            if composite_sim >= threshold:
                redundant_with.append((facets_subcats[j], composite_sim))
        
        df.at[idx, 'Similarity'] = round(best_similarity, 3)
        
        if redundant_with:
            redundant_with_sorted = sorted(redundant_with, key=lambda x: x[1], reverse=True)
            redundant_str = '; '.join([f"{name} ({sim:.2f})" for name, sim in redundant_with_sorted[:3]])
            df.at[idx, 'Redundant'] = redundant_str
    
    for idx in facets_indices:
        df.at[idx, 'Similarity'] = 1.0
        df.at[idx, 'Redundant'] = 'REFERENCE'
    
    df.to_csv(output_path, index=False, quoting=1)
    print(f"Output saved to: {output_path}")
    
    return df


def generate_summary(df: pd.DataFrame) -> str:
    """
    Generate a summary report of redundancy analysis.
    
    Args:
        df: DataFrame with redundancy analysis results
        
    Returns:
        Summary report as a string
    """
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("REDUNDANCY ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    
    non_facets = df[df['Source'] != 'FACETS-60']
    
    for source in sorted(non_facets['Source'].unique()):
        source_df = non_facets[non_facets['Source'] == source]
        redundant_count = (source_df['Redundant'] != '').sum()
        high_sim = (source_df['Similarity'] >= 0.6).sum()
        med_sim = ((source_df['Similarity'] >= 0.35) & (source_df['Similarity'] < 0.6)).sum()
        low_sim = (source_df['Similarity'] < 0.35).sum()
        avg_sim = source_df['Similarity'].mean()
        
        summary_lines.append(f"\n{source}:")
        summary_lines.append(f"  Total items: {len(source_df)}")
        summary_lines.append(f"  Redundant with FACETS (≥0.35): {redundant_count} ({redundant_count/len(source_df)*100:.1f}%)")
        summary_lines.append(f"  High similarity (≥0.6): {high_sim}")
        summary_lines.append(f"  Medium similarity (0.35-0.6): {med_sim}")
        summary_lines.append(f"  Low similarity (<0.35): {low_sim}")
        summary_lines.append(f"  Average similarity: {avg_sim:.3f}")
    
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("TOP REDUNDANT ITEMS BY SOURCE")
    summary_lines.append("=" * 70)
    
    for source in sorted(non_facets['Source'].unique()):
        source_df = non_facets[non_facets['Source'] == source].copy()
        top_redundant = source_df.nlargest(5, 'Similarity')[['Subcategory', 'Similarity', 'Redundant']]
        summary_lines.append(f"\n{source} - Top 5 most similar to FACETS:")
        for _, row in top_redundant.iterrows():
            if row['Similarity'] > 0:
                summary_lines.append(f"  {row['Subcategory']}: {row['Similarity']:.3f} -> {row['Redundant'][:60] if row['Redundant'] else 'None'}")
    
    return '\n'.join(summary_lines)


if __name__ == "__main__":
    input_file = "/mnt/user-data/uploads/facets_-_5__facets__60____A__105____LLM__124____F__523_.csv"
    output_file = "/home/claude/facets_redundancy_analysis.csv"
    
    df = analyze_redundancies(input_file, output_file, threshold=0.35)
    
    summary = generate_summary(df)
    print(summary)
    
    summary_file = "/home/claude/redundancy_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")

