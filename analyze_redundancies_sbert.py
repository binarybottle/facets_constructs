"""
Analyze redundancies using Sentence-BERT embeddings for semantic similarity.

This script compares subcategories from Clinical scales, LLM-generated sources,
and Framework sources against FACETS-60 as the reference set. It computes
semantic similarity using Sentence-BERT embeddings and marks items as redundant
if their similarity exceeds a specified threshold.

Methodology:
    1. Load CSV data with columns: Source, Category, Subcategory, Brief Description, Description
    2. Separate FACETS-60 (reference) items from other items
    3. Create composite text for each item: subcategory (2x weighted) + brief + full description
    4. Generate embeddings using Sentence-BERT (all-mpnet-base-v2)
    5. Compute cosine similarity between each non-FACETS item and all FACETS items
    6. Mark items as redundant if max similarity exceeds threshold

Output:
    - CSV file with added 'Redundant' and 'Similarity' columns
    - PNG file with similarity distribution diagnostics
    - TXT file with summary report

Reference:
    Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks. In Proceedings of EMNLP-IJCNLP, 3982-3992.

Example usage:
    python analyze_redundancies_sbert.py input.csv output.csv --threshold 0.35

    python analyze_redundancies_sbert.py \\
        /path/to/assessments.csv \\
        /path/to/results.csv \\
        --threshold 0.40 \\
        --model all-MiniLM-L6-v2
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(input_path: str) -> pd.DataFrame:
    """
    Load and prepare the input CSV data for analysis.

    Reads a CSV file containing assessment items and standardizes column names.
    Handles missing columns by creating empty placeholders.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        DataFrame with standardized columns: Source, Category, Subcategory,
        Brief Description, Description.

    Raises:
        FileNotFoundError: If input_path does not exist.
        pd.errors.EmptyDataError: If CSV file is empty.
    """
    df = pd.read_csv(input_path, encoding='utf-8', on_bad_lines='skip')
    df.columns = [col.strip() for col in df.columns]

    required_cols = ['Source', 'Category', 'Subcategory', 'Brief Description', 'Description']
    available_cols = [col for col in df.columns if col in required_cols]

    if len(available_cols) < len(required_cols):
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''

    df = df[required_cols].copy()

    df = df.dropna(subset=['Source']).copy()
    df = df[df['Source'].str.strip() != ''].copy()
    df = df.reset_index(drop=True)

    for col in ['Subcategory', 'Brief Description', 'Description']:
        df[col] = df[col].fillna('')

    return df


def create_composite_text(row: pd.Series) -> str:
    """
    Create composite text representation for embedding generation.

    Combines subcategory name (repeated for 2x weighting), brief description,
    and full description into a single text string for embedding.

    Args:
        row: DataFrame row containing Subcategory, Brief Description, Description.

    Returns:
        Composite text string suitable for embedding.
    """
    subcat = str(row['Subcategory']).strip()
    brief = str(row['Brief Description']).strip()
    desc = str(row['Description']).strip()

    composite = f"{subcat}. {subcat}. {brief}. {desc}"
    return composite.strip()


def compute_sbert_similarities(
    model: SentenceTransformer,
    other_texts: list,
    facets_texts: list
) -> np.ndarray:
    """
    Compute Sentence-BERT cosine similarities between two sets of texts.

    Encodes both text sets using the provided model and computes pairwise
    cosine similarities.

    Args:
        model: Pre-trained SentenceTransformer model.
        other_texts: List of text strings to compare (non-FACETS items).
        facets_texts: List of FACETS reference text strings.

    Returns:
        Similarity matrix of shape (len(other_texts), len(facets_texts))
        with values in range [0, 1].
    """
    print("Encoding texts with Sentence-BERT (this may take a minute)...")
    other_embeddings = model.encode(
        other_texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )
    facets_embeddings = model.encode(
        facets_texts,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    print("Computing cosine similarities...")
    similarities = util.cos_sim(other_embeddings, facets_embeddings).cpu().numpy()

    return similarities


def compute_similarity_statistics(max_similarities: np.ndarray) -> dict:
    """
    Compute descriptive statistics for similarity score distribution.

    Args:
        max_similarities: Array of maximum similarity scores per item.

    Returns:
        Dictionary containing mean, median, std, quartiles, and count.
    """
    return {
        'mean': round(float(np.mean(max_similarities)), 3),
        'median': round(float(np.median(max_similarities)), 3),
        'std': round(float(np.std(max_similarities)), 3),
        'q25': round(float(np.percentile(max_similarities, 25)), 3),
        'q75': round(float(np.percentile(max_similarities, 75)), 3),
        'min': round(float(np.min(max_similarities)), 3),
        'max': round(float(np.max(max_similarities)), 3),
        'n_items': len(max_similarities)
    }


def create_diagnostic_plots(
    max_similarities: np.ndarray,
    threshold: float,
    output_path: str
) -> None:
    """
    Create diagnostic visualization plots for similarity analysis.

    Generates a 2x2 figure with:
        - Sorted similarity curve
        - Histogram of similarity scores
        - Cumulative distribution function
        - Box plots by similarity range

    Args:
        max_similarities: Array of maximum similarity scores per item.
        threshold: Similarity threshold for redundancy determination.
        output_path: Path to save the output PNG file.
    """
    sorted_sims = np.sort(max_similarities)[::-1]
    x = np.arange(len(sorted_sims))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sorted similarity curve
    ax1 = axes[0, 0]
    ax1.plot(x, sorted_sims, 'b-', linewidth=2, label='Similarity scores')
    ax1.axhline(
        y=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold={threshold:.3f}'
    )
    ax1.set_xlabel('Item rank (sorted by max similarity)', fontsize=11)
    ax1.set_ylabel('Maximum similarity to FACETS-60', fontsize=11)
    ax1.set_title('Sorted Similarity Curve', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram
    ax2 = axes[0, 1]
    ax2.hist(max_similarities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(
        x=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold={threshold:.3f}'
    )
    ax2.set_xlabel('Maximum similarity score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Similarity Scores', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Cumulative distribution
    ax3 = axes[1, 0]
    sorted_sims_asc = np.sort(max_similarities)
    cumulative = np.arange(1, len(sorted_sims_asc) + 1) / len(sorted_sims_asc) * 100
    ax3.plot(sorted_sims_asc, cumulative, 'g-', linewidth=2)
    ax3.axvline(
        x=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'Threshold={threshold:.3f}'
    )
    pct_below = (max_similarities < threshold).sum() / len(max_similarities) * 100
    ax3.set_xlabel('Maximum similarity score', fontsize=11)
    ax3.set_ylabel('Cumulative percentage', fontsize=11)
    ax3.set_title(
        f'Cumulative Distribution ({pct_below:.1f}% below threshold)',
        fontsize=12,
        fontweight='bold'
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Box plots by similarity range
    ax4 = axes[1, 1]
    ranges = []
    labels = []
    bins = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(len(bins) - 1):
        mask = (max_similarities >= bins[i]) & (max_similarities < bins[i + 1])
        if mask.sum() > 0:
            ranges.append(max_similarities[mask])
            labels.append(f'{bins[i]:.1f}-{bins[i + 1]:.1f}')

    if ranges:
        bp = ax4.boxplot(ranges, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax4.axhline(
            y=threshold,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Threshold={threshold:.3f}'
        )
        ax4.set_xlabel('Similarity range', fontsize=11)
        ax4.set_ylabel('Similarity score', fontsize=11)
        ax4.set_title('Distribution by Similarity Range', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostic plots saved to: {output_path}")
    plt.close()


def analyze_with_sbert(
    input_path: str,
    output_path: str,
    model_name: str = 'all-mpnet-base-v2',
    threshold: float = 0.35
) -> tuple:
    """
    Analyze redundancies using Sentence-BERT embeddings.

    Main analysis function that orchestrates data loading, embedding generation,
    similarity computation, and result annotation.

    Args:
        input_path: Path to input CSV file containing assessment items.
        output_path: Path for output CSV file with redundancy annotations.
        model_name: Name of Sentence-BERT model from HuggingFace.
            Default: 'all-mpnet-base-v2' (recommended for semantic similarity).
        threshold: Similarity threshold above which items are marked redundant.
            Default: 0.35.

    Returns:
        Tuple of (annotated DataFrame, statistics dictionary).
    """
    print("=" * 70)
    print("REDUNDANCY ANALYSIS WITH SENTENCE-BERT")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print(f"Reference: Reimers & Gurevych (2019)")
    print()

    print("Loading data...")
    df = load_and_prepare_data(input_path)

    facets_mask = df['Source'] == 'FACETS-60'
    facets_df = df[facets_mask].copy()
    other_df = df[~facets_mask].copy()

    print(f"FACETS-60 rows: {len(facets_df)}")
    print(f"Other rows: {len(other_df)}")
    print(f"Sources: {list(df['Source'].unique())}")
    print()

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded: {model_name}")
    print()

    print("Creating composite text representations...")
    other_texts = other_df.apply(create_composite_text, axis=1).tolist()
    facets_texts = facets_df.apply(create_composite_text, axis=1).tolist()
    facets_subcats = facets_df['Subcategory'].tolist()

    similarity_matrix = compute_sbert_similarities(model, other_texts, facets_texts)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print()

    max_similarities = similarity_matrix.max(axis=1)
    stats = compute_similarity_statistics(max_similarities)
    stats['threshold'] = threshold

    print("Similarity distribution statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  IQR: [{stats['q25']:.3f}, {stats['q75']:.3f}]")
    print()

    plot_path = str(Path(output_path).with_suffix('')) + '_diagnostics.png'
    create_diagnostic_plots(max_similarities, threshold, plot_path)
    print()

    df['Redundant'] = ''
    df['Similarity'] = 0.0

    print("Annotating redundancies...")
    for i, idx in enumerate(other_df.index):
        max_sim = similarity_matrix[i].max()
        df.at[idx, 'Similarity'] = round(float(max_sim), 3)

        above_threshold = np.where(similarity_matrix[i] >= threshold)[0]

        if len(above_threshold) > 0:
            redundant_pairs = [
                (facets_subcats[j], similarity_matrix[i, j])
                for j in above_threshold
            ]
            redundant_pairs.sort(key=lambda x: x[1], reverse=True)
            redundant_str = '; '.join([
                f"{name} ({sim:.2f})"
                for name, sim in redundant_pairs[:3]
            ])
            df.at[idx, 'Redundant'] = redundant_str

    for idx in facets_df.index:
        df.at[idx, 'Similarity'] = 1.0
        df.at[idx, 'Redundant'] = 'REFERENCE'

    df.to_csv(output_path, index=False, quoting=1)
    print(f"Output saved to: {output_path}")
    print()

    return df, stats


def generate_summary(df: pd.DataFrame, threshold: float) -> str:
    """
    Generate a summary report of the redundancy analysis.

    Creates a formatted text report including per-source statistics,
    top redundant items, and methodology notes.

    Args:
        df: DataFrame with redundancy analysis results.
        threshold: Similarity threshold used for redundancy determination.

    Returns:
        Formatted summary report as a string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("REDUNDANCY ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Method: Sentence-BERT (Reimers & Gurevych, 2019)")
    lines.append(f"Model: all-mpnet-base-v2")
    lines.append(f"Threshold: {threshold}")
    lines.append("=" * 70)

    non_facets = df[df['Source'] != 'FACETS-60']

    for source in sorted(non_facets['Source'].unique()):
        source_df = non_facets[non_facets['Source'] == source]
        n_items = len(source_df)

        if n_items == 0:
            lines.append(f"\n{source}: No items")
            continue

        redundant_count = (source_df['Redundant'] != '').sum()
        high_sim = (source_df['Similarity'] >= 0.6).sum()
        med_sim = ((source_df['Similarity'] >= threshold) & (source_df['Similarity'] < 0.6)).sum()
        low_sim = (source_df['Similarity'] < threshold).sum()
        avg_sim = source_df['Similarity'].mean()

        pct_redundant = redundant_count / n_items * 100

        lines.append(f"\n{source}:")
        lines.append(f"  Total items: {n_items}")
        lines.append(f"  Redundant with FACETS (>={threshold}): {redundant_count} ({pct_redundant:.1f}%)")
        lines.append(f"  High similarity (>=0.6): {high_sim}")
        lines.append(f"  Medium similarity ({threshold}-0.6): {med_sim}")
        lines.append(f"  Low similarity (<{threshold}): {low_sim}")
        lines.append(f"  Average similarity: {avg_sim:.3f}")

    lines.append("\n" + "=" * 70)
    lines.append("TOP REDUNDANT ITEMS BY SOURCE")
    lines.append("=" * 70)

    for source in sorted(non_facets['Source'].unique()):
        source_df = non_facets[non_facets['Source'] == source].copy()
        if len(source_df) == 0:
            continue

        top_redundant = source_df.nlargest(5, 'Similarity')[
            ['Subcategory', 'Similarity', 'Redundant']
        ]
        lines.append(f"\n{source} - Top 5 most similar to FACETS:")
        for _, row in top_redundant.iterrows():
            if row['Similarity'] > 0:
                redundant_display = row['Redundant'][:60] if row['Redundant'] else 'None'
                lines.append(
                    f"  {row['Subcategory']}: {row['Similarity']:.3f} -> {redundant_display}"
                )

    lines.append("\n" + "=" * 70)
    lines.append("METHODOLOGY")
    lines.append("=" * 70)
    lines.append("Semantic similarity computed using Sentence-BERT embeddings.")
    lines.append("Model: all-mpnet-base-v2 (768-dimensional embeddings)")
    lines.append("Composite text: subcategory (2x) + brief description + full description")
    lines.append("Similarity metric: Cosine similarity between embedding vectors")
    lines.append("")
    lines.append("Reference:")
    lines.append("  Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence")
    lines.append("  Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP, 3982-3992.")

    return '\n'.join(lines)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Analyze redundancies using Sentence-BERT semantic similarity.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  %(prog)s input.csv output.csv --threshold 0.40
  %(prog)s input.csv output.csv --model all-MiniLM-L6-v2 --threshold 0.35
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file containing assessment items'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path for output CSV file with redundancy annotations'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.35,
        help='Similarity threshold for redundancy (default: 0.35)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all-mpnet-base-v2',
        help='Sentence-BERT model name (default: all-mpnet-base-v2)'
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the redundancy analysis script.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_arguments()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df, stats = analyze_with_sbert(
            str(input_path),
            str(output_path),
            model_name=args.model,
            threshold=args.threshold
        )

        summary = generate_summary(df, args.threshold)
        print("\n" + summary)

        summary_path = str(output_path.with_suffix('')) + '_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_path}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())