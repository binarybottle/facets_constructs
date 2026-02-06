"""
Analyze redundancies using Sentence-BERT embeddings.

This script uses scientifically validated semantic similarity (Reimers & Gurevych, 2019)
to identify redundant assessment subcategories. It compares items against FACETS-60
reference items and also identifies redundancies among non-FACETS items.

Reference:
Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese 
BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural 
Language Processing and the 9th International Joint Conference on Natural Language Processing 
(EMNLP-IJCNLP), 3982–3992.

Author: Generated for assessment subcategory analysis
Date: 2026-02-04
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """Load and prepare the input CSV data."""
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]
    
    keep_cols = ['Source', 'Category', 'Subcategory', 'Brief Description', 'Description']
    available_cols = [col for col in df.columns if col in keep_cols][:5]
    
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


def compute_all_similarities(model: SentenceTransformer, 
                              texts: list) -> np.ndarray:
    """
    Compute Sentence-BERT cosine similarities between all pairs of texts.
    
    Args:
        model: Pre-trained SentenceTransformer model
        texts: List of text strings
        
    Returns:
        Similarity matrix of shape (len(texts), len(texts))
    """
    print("Encoding texts with Sentence-BERT (this may take a minute)...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    
    print("Computing all pairwise cosine similarities...")
    similarities = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    return similarities


def create_diagnostic_plots(max_facets_sims: np.ndarray,
                            max_nonfacets_sims: np.ndarray,
                            threshold: float,
                            plot_output_path: str = None) -> dict:
    """
    Create diagnostic plots for similarity distributions.
    
    Args:
        max_facets_sims: Array of max similarities to FACETS items
        max_nonfacets_sims: Array of max similarities to non-FACETS items
        threshold: Similarity threshold to visualize
        plot_output_path: Optional path to save diagnostic plots
        
    Returns:
        Dictionary with distribution statistics
    """
    # Calculate distribution statistics
    stats = {
        'threshold': round(float(threshold), 3),
        'facets_mean': round(float(np.mean(max_facets_sims)), 3),
        'facets_median': round(float(np.median(max_facets_sims)), 3),
        'facets_std': round(float(np.std(max_facets_sims)), 3),
        'nonfacets_mean': round(float(np.mean(max_nonfacets_sims[max_nonfacets_sims > 0])), 3) if np.any(max_nonfacets_sims > 0) else 0,
        'nonfacets_median': round(float(np.median(max_nonfacets_sims[max_nonfacets_sims > 0])), 3) if np.any(max_nonfacets_sims > 0) else 0,
        'n_items': len(max_facets_sims)
    }
    
    # Create visualization
    if plot_output_path:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: FACETS similarity histogram
        ax1 = axes[0, 0]
        ax1.hist(max_facets_sims, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.3f}')
        ax1.set_xlabel('Maximum similarity to FACETS-60', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title('Distribution of Max FACETS Similarities', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Non-FACETS similarity histogram
        ax2 = axes[0, 1]
        nonfacets_positive = max_nonfacets_sims[max_nonfacets_sims > 0]
        if len(nonfacets_positive) > 0:
            ax2.hist(nonfacets_positive, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.3f}')
        ax2.set_xlabel('Maximum similarity to non-FACETS items', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Max Non-FACETS Similarities', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Sorted FACETS similarity curve
        ax3 = axes[1, 0]
        sorted_facets = np.sort(max_facets_sims)[::-1]
        x = np.arange(len(sorted_facets))
        ax3.plot(x, sorted_facets, 'b-', linewidth=2)
        ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold={threshold:.3f}')
        ax3.set_xlabel('Item rank (sorted by max similarity)', fontsize=11)
        ax3.set_ylabel('Maximum similarity to FACETS-60', fontsize=11)
        ax3.set_title('Sorted FACETS Similarity Curve', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot FACETS vs non-FACETS similarity
        ax4 = axes[1, 1]
        ax4.scatter(max_facets_sims, max_nonfacets_sims, alpha=0.5, s=30)
        ax4.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax4.set_xlabel('Max FACETS similarity', fontsize=11)
        ax4.set_ylabel('Max non-FACETS similarity', fontsize=11)
        ax4.set_title('FACETS vs Non-FACETS Similarity', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plots saved to: {plot_output_path}")
        plt.close()
    
    return stats


def analyze_with_sbert(input_path: str, 
                       output_path: str,
                       model_name: str = 'all-mpnet-base-v2',
                       threshold: float = 0.35) -> tuple:
    """
    Analyze redundancies using Sentence-BERT embeddings.
    
    Generates a CSV with 6 new columns:
    1. Max FACETS similarity value - max similarity to any FACETS row
    2. Max FACETS similarity - subcategory name with max FACETS similarity
    3. Max non-FACETS similarity value - max similarity to any non-FACETS row
    4. Max non-FACETS similarity - subcategory name with max non-FACETS similarity
    5. Redundant FACETS - whether redundant with any FACETS row (>= threshold)
    6. Redundant non-FACETS - whether redundant with any non-FACETS row (>= threshold)
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        model_name: Name of Sentence-BERT model to use
        threshold: Similarity threshold above which items are marked as redundant
        
    Returns:
        Tuple of (DataFrame with results, statistics dict)
    """
    print("=" * 70)
    print("REDUNDANCY ANALYSIS WITH SENTENCE-BERT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Reference: Reimers & Gurevych (2019)")
    print(f"Threshold: {threshold}")
    print()
    
    # Load data
    print("Loading data...")
    df = load_and_prepare_data(input_path)
    
    # Identify FACETS and non-FACETS items
    facets_mask = df['Source'] == 'FACETS-60'
    facets_indices = df[facets_mask].index.tolist()
    other_indices = df[~facets_mask].index.tolist()
    
    print(f"Total rows: {len(df)}")
    print(f"FACETS-60 rows: {len(facets_indices)}")
    print(f"Non-FACETS rows: {len(other_indices)}")
    print(f"Sources: {df['Source'].unique().tolist()}")
    print()
    
    # Load Sentence-BERT model
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded: {model_name}")
    print()
    
    # Create composite texts for all items
    print("Creating composite text representations...")
    all_texts = df.apply(create_composite_text, axis=1).tolist()
    all_subcats = df['Subcategory'].tolist()
    
    # Compute all pairwise similarities
    similarity_matrix = compute_all_similarities(model, all_texts)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print()
    
    # Initialize new columns
    df['Max FACETS similarity value'] = 0.0
    df['Max FACETS similarity'] = ''
    df['Max non-FACETS similarity value'] = 0.0
    df['Max non-FACETS similarity'] = ''
    df['Redundant FACETS'] = ''
    df['Redundant non-FACETS'] = ''
    
    print("Computing similarities and annotating redundancies...")
    
    # Arrays to collect statistics
    max_facets_sims = []
    max_nonfacets_sims = []
    
    for i in range(len(df)):
        row_source = df.at[i, 'Source']
        is_facets = (row_source == 'FACETS-60')
        
        # --- FACETS similarity ---
        if is_facets:
            # For FACETS items, find max similarity to OTHER FACETS items (exclude self)
            facets_sims = []
            for j in facets_indices:
                if j != i:  # Exclude self
                    facets_sims.append((j, similarity_matrix[i, j]))
            
            if facets_sims:
                best_facets = max(facets_sims, key=lambda x: x[1])
                best_facets_idx, best_facets_sim = best_facets
                df.at[i, 'Max FACETS similarity value'] = round(float(best_facets_sim), 3)
                df.at[i, 'Max FACETS similarity'] = all_subcats[best_facets_idx]
                max_facets_sims.append(best_facets_sim)
                
                if best_facets_sim >= threshold:
                    df.at[i, 'Redundant FACETS'] = 'Yes'
                else:
                    df.at[i, 'Redundant FACETS'] = 'No'
            else:
                df.at[i, 'Redundant FACETS'] = 'No'
                max_facets_sims.append(0.0)
        else:
            # For non-FACETS items, find max similarity to any FACETS item
            facets_sims = [(j, similarity_matrix[i, j]) for j in facets_indices]
            
            if facets_sims:
                best_facets = max(facets_sims, key=lambda x: x[1])
                best_facets_idx, best_facets_sim = best_facets
                df.at[i, 'Max FACETS similarity value'] = round(float(best_facets_sim), 3)
                df.at[i, 'Max FACETS similarity'] = all_subcats[best_facets_idx]
                max_facets_sims.append(best_facets_sim)
                
                if best_facets_sim >= threshold:
                    df.at[i, 'Redundant FACETS'] = 'Yes'
                else:
                    df.at[i, 'Redundant FACETS'] = 'No'
            else:
                df.at[i, 'Redundant FACETS'] = 'No'
                max_facets_sims.append(0.0)
        
        # --- Non-FACETS similarity ---
        if is_facets:
            # For FACETS items, find max similarity to any non-FACETS item
            nonfacets_sims = [(j, similarity_matrix[i, j]) for j in other_indices]
            
            if nonfacets_sims:
                best_nonfacets = max(nonfacets_sims, key=lambda x: x[1])
                best_nonfacets_idx, best_nonfacets_sim = best_nonfacets
                df.at[i, 'Max non-FACETS similarity value'] = round(float(best_nonfacets_sim), 3)
                df.at[i, 'Max non-FACETS similarity'] = all_subcats[best_nonfacets_idx]
                max_nonfacets_sims.append(best_nonfacets_sim)
                
                if best_nonfacets_sim >= threshold:
                    df.at[i, 'Redundant non-FACETS'] = 'Yes'
                else:
                    df.at[i, 'Redundant non-FACETS'] = 'No'
            else:
                df.at[i, 'Redundant non-FACETS'] = 'No'
                max_nonfacets_sims.append(0.0)
        else:
            # For non-FACETS items, find max similarity to OTHER non-FACETS items (exclude self)
            nonfacets_sims = []
            for j in other_indices:
                if j != i:  # Exclude self
                    nonfacets_sims.append((j, similarity_matrix[i, j]))
            
            if nonfacets_sims:
                best_nonfacets = max(nonfacets_sims, key=lambda x: x[1])
                best_nonfacets_idx, best_nonfacets_sim = best_nonfacets
                df.at[i, 'Max non-FACETS similarity value'] = round(float(best_nonfacets_sim), 3)
                df.at[i, 'Max non-FACETS similarity'] = all_subcats[best_nonfacets_idx]
                max_nonfacets_sims.append(best_nonfacets_sim)
                
                if best_nonfacets_sim >= threshold:
                    df.at[i, 'Redundant non-FACETS'] = 'Yes'
                else:
                    df.at[i, 'Redundant non-FACETS'] = 'No'
            else:
                df.at[i, 'Redundant non-FACETS'] = 'No'
                max_nonfacets_sims.append(0.0)
    
    max_facets_sims = np.array(max_facets_sims)
    max_nonfacets_sims = np.array(max_nonfacets_sims)
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    plot_path = output_path.replace('.csv', '_diagnostic_plots.png')
    stats = create_diagnostic_plots(max_facets_sims, max_nonfacets_sims, threshold, 
                                    plot_output_path=plot_path)
    
    print(f"\nSimilarity distribution:")
    print(f"  FACETS - Mean: {stats['facets_mean']:.3f}, Median: {stats['facets_median']:.3f}")
    print(f"  Non-FACETS - Mean: {stats['nonfacets_mean']:.3f}, Median: {stats['nonfacets_median']:.3f}")
    print()
    
    # Save output
    df.to_csv(output_path, index=False, quoting=1)
    print(f"Output saved to: {output_path}")
    print()
    
    return df, stats


def generate_summary(df: pd.DataFrame, threshold: float, method: str = "Sentence-BERT") -> str:
    """Generate summary report with methodology."""
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("REDUNDANCY ANALYSIS SUMMARY")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Method: {method} (Reimers & Gurevych, 2019)")
    summary_lines.append(f"Threshold: {threshold}")
    summary_lines.append("=" * 70)
    
    # Overall statistics
    total_items = len(df)
    facets_items = len(df[df['Source'] == 'FACETS-60'])
    non_facets_items = total_items - facets_items
    
    summary_lines.append(f"\nTotal items: {total_items}")
    summary_lines.append(f"FACETS-60 items: {facets_items}")
    summary_lines.append(f"Non-FACETS items: {non_facets_items}")
    
    # Redundancy counts
    redundant_facets = (df['Redundant FACETS'] == 'Yes').sum()
    redundant_nonfacets = (df['Redundant non-FACETS'] == 'Yes').sum()
    
    summary_lines.append(f"\nItems redundant with FACETS (>={threshold}): {redundant_facets} ({redundant_facets/total_items*100:.1f}%)")
    summary_lines.append(f"Items redundant with non-FACETS (>={threshold}): {redundant_nonfacets} ({redundant_nonfacets/total_items*100:.1f}%)")
    
    # Per-source breakdown
    summary_lines.append("\n" + "-" * 70)
    summary_lines.append("BREAKDOWN BY SOURCE")
    summary_lines.append("-" * 70)
    
    for source in sorted(df['Source'].unique()):
        source_df = df[df['Source'] == source]
        total = len(source_df)
        
        if total == 0:
            continue
            
        red_facets = (source_df['Redundant FACETS'] == 'Yes').sum()
        red_nonfacets = (source_df['Redundant non-FACETS'] == 'Yes').sum()
        avg_facets_sim = source_df['Max FACETS similarity value'].mean()
        avg_nonfacets_sim = source_df['Max non-FACETS similarity value'].mean()
        
        summary_lines.append(f"\n{source}:")
        summary_lines.append(f"  Total items: {total}")
        summary_lines.append(f"  Redundant with FACETS: {red_facets} ({red_facets/total*100:.1f}%)")
        summary_lines.append(f"  Redundant with non-FACETS: {red_nonfacets} ({red_nonfacets/total*100:.1f}%)")
        summary_lines.append(f"  Avg FACETS similarity: {avg_facets_sim:.3f}")
        summary_lines.append(f"  Avg non-FACETS similarity: {avg_nonfacets_sim:.3f}")
    
    # High similarity items
    summary_lines.append("\n" + "-" * 70)
    summary_lines.append("TOP 10 ITEMS MOST SIMILAR TO FACETS-60")
    summary_lines.append("-" * 70)
    
    non_facets = df[df['Source'] != 'FACETS-60'].copy()
    if len(non_facets) > 0:
        top_facets = non_facets.nlargest(10, 'Max FACETS similarity value')
        for _, row in top_facets.iterrows():
            summary_lines.append(f"  {row['Subcategory']} ({row['Source']})")
            summary_lines.append(f"    -> {row['Max FACETS similarity']} ({row['Max FACETS similarity value']:.3f})")
    
    summary_lines.append("\n" + "-" * 70)
    summary_lines.append("TOP 10 ITEMS MOST SIMILAR TO OTHER NON-FACETS")
    summary_lines.append("-" * 70)
    
    if len(non_facets) > 0:
        top_nonfacets = non_facets.nlargest(10, 'Max non-FACETS similarity value')
        for _, row in top_nonfacets.iterrows():
            summary_lines.append(f"  {row['Subcategory']} ({row['Source']})")
            summary_lines.append(f"    -> {row['Max non-FACETS similarity']} ({row['Max non-FACETS similarity value']:.3f})")
    
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("METHODOLOGY NOTE")
    summary_lines.append("=" * 70)
    summary_lines.append("Similarity computed using Sentence-BERT embeddings (all-mpnet-base-v2).")
    summary_lines.append("Reference: Reimers, N., & Gurevych, I. (2019). Sentence-BERT:")
    summary_lines.append("  Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP.")
    summary_lines.append(f"Threshold: {threshold}")
    summary_lines.append("")
    summary_lines.append("Columns added:")
    summary_lines.append("  - Max FACETS similarity value: Maximum similarity to any FACETS-60 item")
    summary_lines.append("  - Max FACETS similarity: Subcategory with maximum FACETS similarity")
    summary_lines.append("  - Max non-FACETS similarity value: Maximum similarity to any non-FACETS item")
    summary_lines.append("  - Max non-FACETS similarity: Subcategory with maximum non-FACETS similarity")
    summary_lines.append("  - Redundant FACETS: 'Yes' if max FACETS similarity >= threshold")
    summary_lines.append("  - Redundant non-FACETS: 'Yes' if max non-FACETS similarity >= threshold")
    
    return '\n'.join(summary_lines)


if __name__ == "__main__":
    import sys
    
    # Default paths and threshold
    default_input = "data/facets - 5. all items.csv"
    default_output = "data/facets - 5. all items - redundancy.csv"
    default_threshold = 0.35
    
    # Usage: python analyze_redundancies_sbert.py [input_file] [output_file] [threshold]
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) >= 4 else default_threshold
    elif len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python analyze_redundancies_sbert.py [input_file] [output_file] [threshold]")
        print(f"\nDefaults:")
        print(f"  input_file:  {default_input}")
        print(f"  output_file: {default_output}")
        print(f"  threshold:   {default_threshold}")
        print("\nExample:")
        print("  python analyze_redundancies_sbert.py data/items.csv data/items_redundancy.csv 0.4")
        sys.exit(0)
    else:
        input_file = default_input
        output_file = default_output
        threshold = default_threshold
        print(f"Using default paths (run with -h for help):")
        print(f"  Input:  {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Threshold: {threshold}")
        print()
    
    # Run analysis
    df, stats = analyze_with_sbert(
        input_file, 
        output_file,
        model_name='all-mpnet-base-v2',
        threshold=threshold
    )
    
    # Generate and save summary
    summary = generate_summary(df, threshold, method="Sentence-BERT")
    print("\n" + summary)
    
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")
