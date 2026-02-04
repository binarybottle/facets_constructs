"""
Analyze redundancies using Sentence-BERT embeddings and elbow method for threshold determination.

This script uses scientifically validated semantic similarity (Reimers & Gurevych, 2019)
and data-driven threshold selection via the elbow method.

Reference:
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese 
BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural 
Language Processing and the 9th International Joint Conference on Natural Language Processing 
(EMNLP-IJCNLP), 3982–3992.

Author: Generated for assessment subcategory analysis
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """Load and prepare the input CSV data."""
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
    
    return df


def create_composite_text(row: pd.Series) -> str:
    """
    Create composite text representation for embedding.
    
    Combines subcategory name (weighted 2x), brief description, and full description.
    """
    subcat = str(row['Subcategory']).strip()
    brief = str(row['Brief Description']).strip()
    desc = str(row['Description']).strip()
    
    # Weight subcategory name by repeating it to increase its importance
    composite = f"{subcat}. {subcat}. {brief}. {desc}"
    return composite.strip()


def compute_sbert_similarities(model: SentenceTransformer, 
                                other_texts: list, 
                                facets_texts: list) -> np.ndarray:
    """
    Compute Sentence-BERT cosine similarities between two sets of texts.
    
    Args:
        model: Pre-trained SentenceTransformer model
        other_texts: List of text strings to compare
        facets_texts: List of FACETS reference texts
        
    Returns:
        Similarity matrix of shape (len(other_texts), len(facets_texts))
    """
    print("Encoding texts with Sentence-BERT (this may take a minute)...")
    other_embeddings = model.encode(other_texts, convert_to_tensor=True, show_progress_bar=True)
    facets_embeddings = model.encode(facets_texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing cosine similarities...")
    similarities = util.cos_sim(other_embeddings, facets_embeddings).cpu().numpy()
    
    return similarities


def find_elbow_threshold(similarities: np.ndarray, 
                         plot_output_path: str = None) -> dict:
    """
    Use elbow method to determine optimal similarity threshold.
    
    Args:
        similarities: Matrix of similarity scores
        plot_output_path: Optional path to save elbow plot
        
    Returns:
        Dictionary with threshold, elbow point info, and statistics
    """
    # Get maximum similarity for each item
    max_sims = similarities.max(axis=1)
    
    # Sort similarities for elbow detection
    sorted_sims = np.sort(max_sims)[::-1]  # Descending order
    
    # Create x-axis (item index)
    x = np.arange(len(sorted_sims))
    
    # Use KneeLocator to find elbow
    # curve='convex' because we expect diminishing returns (high sims first, then drops off)
    # direction='decreasing' because similarities decrease as we go down the sorted list
    try:
        kneedle = KneeLocator(x, sorted_sims, 
                             curve='convex', 
                             direction='decreasing',
                             online=True)
        elbow_idx = kneedle.elbow
        elbow_threshold = sorted_sims[elbow_idx] if elbow_idx is not None else 0.35
    except Exception as e:
        print(f"Elbow detection warning: {e}")
        # Fallback: use median or 75th percentile
        elbow_threshold = np.percentile(max_sims, 75)
        elbow_idx = np.searchsorted(-sorted_sims, -elbow_threshold)
    
    # Calculate distribution statistics
    stats = {
        'threshold': round(float(elbow_threshold), 3),
        'elbow_index': int(elbow_idx) if elbow_idx is not None else 0,
        'mean': float(np.mean(max_sims)),
        'median': float(np.median(max_sims)),
        'std': float(np.std(max_sims)),
        'q25': float(np.percentile(max_sims, 25)),
        'q75': float(np.percentile(max_sims, 75)),
        'n_items': len(max_sims)
    }
    
    # Create visualization
    if plot_output_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Elbow curve
        ax1 = axes[0, 0]
        ax1.plot(x, sorted_sims, 'b-', linewidth=2, label='Similarity scores')
        if elbow_idx is not None:
            ax1.axvline(x=elbow_idx, color='red', linestyle='--', linewidth=2, 
                       label=f'Elbow point (threshold={elbow_threshold:.3f})')
            ax1.axhline(y=elbow_threshold, color='red', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Item rank (sorted by max similarity)', fontsize=11)
        ax1.set_ylabel('Maximum similarity to FACETS-60', fontsize=11)
        ax1.set_title('Elbow Method: Sorted Similarity Curve', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        ax2 = axes[0, 1]
        ax2.hist(max_sims, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=elbow_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={elbow_threshold:.3f}')
        ax2.set_xlabel('Maximum similarity score', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Similarity Scores', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Cumulative distribution
        ax3 = axes[1, 0]
        sorted_sims_asc = np.sort(max_sims)
        cumulative = np.arange(1, len(sorted_sims_asc) + 1) / len(sorted_sims_asc) * 100
        ax3.plot(sorted_sims_asc, cumulative, 'g-', linewidth=2)
        ax3.axvline(x=elbow_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={elbow_threshold:.3f}')
        pct_below = (max_sims < elbow_threshold).sum() / len(max_sims) * 100
        ax3.set_xlabel('Maximum similarity score', fontsize=11)
        ax3.set_ylabel('Cumulative percentage', fontsize=11)
        ax3.set_title(f'Cumulative Distribution ({pct_below:.1f}% below threshold)', 
                     fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Box plot by similarity range
        ax4 = axes[1, 1]
        ranges = []
        labels = []
        bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            mask = (max_sims >= bins[i]) & (max_sims < bins[i+1])
            if mask.sum() > 0:
                ranges.append(max_sims[mask])
                labels.append(f'{bins[i]:.1f}-{bins[i+1]:.1f}')
        
        bp = ax4.boxplot(ranges, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax4.axhline(y=elbow_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={elbow_threshold:.3f}')
        ax4.set_xlabel('Similarity range', fontsize=11)
        ax4.set_ylabel('Similarity score', fontsize=11)
        ax4.set_title('Distribution by Similarity Range', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        print(f"Elbow plot saved to: {plot_output_path}")
        plt.close()
    
    return stats


def analyze_with_sbert(input_path: str, 
                       output_path: str,
                       model_name: str = 'all-mpnet-base-v2',
                       auto_threshold: bool = True,
                       manual_threshold: float = None) -> tuple:
    """
    Analyze redundancies using Sentence-BERT embeddings.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        model_name: Name of Sentence-BERT model to use
        auto_threshold: If True, use elbow method to determine threshold
        manual_threshold: Manual threshold to use if auto_threshold is False
        
    Returns:
        Tuple of (DataFrame with results, threshold statistics dict)
    """
    print("=" * 70)
    print("REDUNDANCY ANALYSIS WITH SENTENCE-BERT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Reference: Reimers & Gurevych (2019)")
    print()
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(input_path)
    
    facets_mask = df['Source'] == 'FACETS-60'
    facets_df = df[facets_mask].copy()
    other_df = df[~facets_mask].copy()
    
    print(f"FACETS-60 rows: {len(facets_df)}")
    print(f"Other rows: {len(other_df)}")
    print(f"Sources: {df['Source'].unique()}")
    print()
    
    # Load Sentence-BERT model
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded: {model_name}")
    print()
    
    # Create composite texts
    print("Creating composite text representations...")
    other_texts = other_df.apply(create_composite_text, axis=1).tolist()
    facets_texts = facets_df.apply(create_composite_text, axis=1).tolist()
    facets_subcats = facets_df['Subcategory'].tolist()
    
    # Compute similarities
    similarity_matrix = compute_sbert_similarities(model, other_texts, facets_texts)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print()
    
    # Determine threshold
    if auto_threshold:
        print("Determining optimal threshold using elbow method...")
        plot_path = output_path.replace('.csv', '_elbow_analysis.png')
        threshold_stats = find_elbow_threshold(similarity_matrix, plot_output_path=plot_path)
        threshold = threshold_stats['threshold']
        print(f"\nElbow method results:")
        print(f"  Optimal threshold: {threshold}")
        print(f"  Mean similarity: {threshold_stats['mean']:.3f}")
        print(f"  Median similarity: {threshold_stats['median']:.3f}")
        print(f"  Std deviation: {threshold_stats['std']:.3f}")
        print(f"  Q1-Q3: [{threshold_stats['q25']:.3f}, {threshold_stats['q75']:.3f}]")
    else:
        threshold = manual_threshold if manual_threshold is not None else 0.35
        threshold_stats = {'threshold': threshold, 'method': 'manual'}
        print(f"Using manual threshold: {threshold}")
    
    print()
    
    # Annotate dataframe with results
    df['Redundant'] = ''
    df['Similarity'] = 0.0
    
    print("Annotating redundancies...")
    for i, idx in enumerate(other_df.index):
        max_sim = similarity_matrix[i].max()
        df.at[idx, 'Similarity'] = round(float(max_sim), 3)
        
        # Find all FACETS items above threshold
        above_threshold = np.where(similarity_matrix[i] >= threshold)[0]
        
        if len(above_threshold) > 0:
            redundant_pairs = [(facets_subcats[j], similarity_matrix[i, j]) 
                              for j in above_threshold]
            redundant_pairs.sort(key=lambda x: x[1], reverse=True)
            redundant_str = '; '.join([f"{name} ({sim:.2f})" 
                                      for name, sim in redundant_pairs[:3]])
            df.at[idx, 'Redundant'] = redundant_str
    
    # Mark FACETS items as reference
    for idx in facets_df.index:
        df.at[idx, 'Similarity'] = 1.0
        df.at[idx, 'Redundant'] = 'REFERENCE'
    
    # Save output
    df.to_csv(output_path, index=False, quoting=1)
    print(f"Output saved to: {output_path}")
    print()
    
    return df, threshold_stats


def generate_summary(df: pd.DataFrame, threshold: float, method: str = "Sentence-BERT") -> str:
    """Generate summary report with methodology."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("REDUNDANCY ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Method: {method} (Reimers & Gurevych, 2019)")
    summary_lines.append(f"Threshold: {threshold} (determined via elbow method)")
    summary_lines.append("=" * 70)
    
    non_facets = df[df['Source'] != 'FACETS-60']
    
    for source in sorted(non_facets['Source'].unique()):
        source_df = non_facets[non_facets['Source'] == source]
        redundant_count = (source_df['Redundant'] != '').sum()
        high_sim = (source_df['Similarity'] >= 0.6).sum()
        med_sim = ((source_df['Similarity'] >= threshold) & (source_df['Similarity'] < 0.6)).sum()
        low_sim = (source_df['Similarity'] < threshold).sum()
        avg_sim = source_df['Similarity'].mean()
        
        summary_lines.append(f"\n{source}:")
        summary_lines.append(f"  Total items: {len(source_df)}")
        summary_lines.append(f"  Redundant with FACETS (≥{threshold}): {redundant_count} ({redundant_count/len(source_df)*100:.1f}%)")
        summary_lines.append(f"  High similarity (≥0.6): {high_sim}")
        summary_lines.append(f"  Medium similarity ({threshold}-0.6): {med_sim}")
        summary_lines.append(f"  Low similarity (<{threshold}): {low_sim}")
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
    
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("METHODOLOGY NOTE")
    summary_lines.append("=" * 70)
    summary_lines.append("Similarity computed using Sentence-BERT embeddings (all-mpnet-base-v2).")
    summary_lines.append("Reference: Reimers, N., & Gurevych, I. (2019). Sentence-BERT:")
    summary_lines.append("  Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.")
    summary_lines.append("Threshold determined via elbow method on similarity distribution.")
    
    return '\n'.join(summary_lines)


if __name__ == "__main__":
    # Update these paths to match your actual file locations
    input_file = "/Users/arno/Downloads/facets - 5. facets (60) + A (105) + LLM (124) + F (523).csv"
    output_file = "/Users/arno/Downloads/facets_redundancy_sbert.csv"
    
    # Run analysis with automatic threshold detection
    df, threshold_stats = analyze_with_sbert(
        input_file, 
        output_file,
        model_name='all-mpnet-base-v2',
        auto_threshold=True
    )
    
    # Generate and save summary
    summary = generate_summary(df, threshold_stats['threshold'], method="Sentence-BERT + Elbow Method")
    print("\n" + summary)
    
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")
