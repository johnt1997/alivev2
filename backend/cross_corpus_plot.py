"""
Cross-Corpus Comparison Plot for Thesis Figure 5.6
Compares Washington vs EU corpus performance across all 12 pipelines
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === CONFIGURATION ===
WA_RESULTS_DIR = Path("./results/Washington/final_run_42Q")
EU_RESULTS_DIR = Path("./results/Eu/final_run")
OUTPUT_PATH = Path("./results/cross_corpus_comparison.png")

# Pipelines in display order (grouped by chunking strategy)
PIPELINE_ORDER = [
    'dense_recursive', 'dense_recursive_rerank',
    'dense_sentence', 'dense_sentence_rerank',
    'dense_semantic', 'dense_semantic_rerank',
    'hybrid_recursive', 'hybrid_recursive_rerank',
    'hybrid_sentence', 'hybrid_sentence_rerank',
    'hybrid_semantic', 'hybrid_semantic_rerank'
]

# Short names for x-axis
SHORT_NAMES = {
    'dense_recursive': 'D-Rec',
    'dense_recursive_rerank': 'D-Rec-R',
    'dense_sentence': 'D-Sent',
    'dense_sentence_rerank': 'D-Sent-R',
    'dense_semantic': 'D-Sem',
    'dense_semantic_rerank': 'D-Sem-R',
    'hybrid_recursive': 'H-Rec',
    'hybrid_recursive_rerank': 'H-Rec-R',
    'hybrid_sentence': 'H-Sent',
    'hybrid_sentence_rerank': 'H-Sent-R',
    'hybrid_semantic': 'H-Sem',
    'hybrid_semantic_rerank': 'H-Sem-R'
}


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load and combine all pipeline CSV results from a directory."""
    all_dfs = []
    for csv_file in results_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        # Extract pipeline name from filename
        pipeline_name = csv_file.stem.replace("_final_results", "")
        df['pipeline'] = pipeline_name
        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    return pd.concat(all_dfs, ignore_index=True)


def compute_pipeline_means(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean Faithfulness per pipeline."""
    return df.groupby('pipeline')['faithfulness'].mean().reset_index()


def create_cross_corpus_plot():
    """Create grouped bar chart comparing WA vs EU Faithfulness."""

    # Load data
    print("Loading Washington results...")
    wa_df = load_all_results(WA_RESULTS_DIR)
    wa_means = compute_pipeline_means(wa_df)
    wa_means = wa_means.set_index('pipeline')

    print("Loading EU results...")
    eu_df = load_all_results(EU_RESULTS_DIR)
    eu_means = compute_pipeline_means(eu_df)
    eu_means = eu_means.set_index('pipeline')

    # Prepare data in correct order
    pipelines = [p for p in PIPELINE_ORDER if p in wa_means.index and p in eu_means.index]
    wa_values = [wa_means.loc[p, 'faithfulness'] for p in pipelines]
    eu_values = [eu_means.loc[p, 'faithfulness'] for p in pipelines]
    labels = [SHORT_NAMES.get(p, p) for p in pipelines]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(pipelines))
    width = 0.35

    bars_wa = ax.bar(x - width/2, wa_values, width, label='Washington', color='#2E86AB', alpha=0.85)
    bars_eu = ax.bar(x + width/2, eu_values, width, label='EU', color='#F18F01', alpha=0.85)

    # Styling
    ax.set_xlabel('Pipeline Configuration', fontsize=11)
    ax.set_ylabel('Faithfulness Score', fontsize=11)
    ax.set_title('Cross-Corpus Comparison: Faithfulness by Pipeline', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Add value labels on bars
    for bar in bars_wa:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color='#2E86AB')

    for bar in bars_eu:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color='#F18F01')

    # Add chunking strategy separators
    for i in [2, 4, 6, 8, 10]:
        if i < len(pipelines):
            ax.axvline(x=i - 0.5, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.5)

    # Add chunking labels at top
    chunk_positions = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5]
    chunk_labels = ['Recursive', 'Sentence', 'Semantic', 'Recursive', 'Sentence', 'Semantic']
    for pos, label in zip(chunk_positions[:min(len(chunk_positions), len(pipelines)//2)], chunk_labels):
        if pos < len(pipelines):
            ax.text(pos, 0.98, label, ha='center', va='bottom', fontsize=8,
                    color='gray', style='italic', transform=ax.get_xaxis_transform())

    plt.tight_layout()

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved to: {OUTPUT_PATH}")

    # Also show
    plt.show()

    # Print summary stats
    print("\n=== Summary Statistics ===")
    print(f"Washington Mean F: {np.mean(wa_values):.3f}")
    print(f"EU Mean F: {np.mean(eu_values):.3f}")
    print(f"\nBest WA: {pipelines[np.argmax(wa_values)]} ({max(wa_values):.3f})")
    print(f"Best EU: {pipelines[np.argmax(eu_values)]} ({max(eu_values):.3f})")


if __name__ == "__main__":
    create_cross_corpus_plot()
