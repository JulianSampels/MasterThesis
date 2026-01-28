import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from collections import defaultdict
import pandas as pd  # Added for easy CSV saving from list of dicts

def create_heatmaps(results, save_dir="./figures"):
    """
    Create heatmaps for total coverage and average recall per group from grid search results.
    Also saves the results to a CSV file.

    Args:
        results: List of dicts with keys 'alpha', 'beta', 'temp', 'total_cov', 'avg_cov_per_group'
        save_dir: Directory to save the PNG images and results file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results to CSV using pandas for simplicity
    df = pd.DataFrame(results)
    df.to_csv(f'{save_dir}/heatmaps_results.csv', index=False)
    
    # Extract unique alphas and betas
    alpha_grid = sorted(set([r['alpha'] for r in results]))  # Unique alphas
    beta_grid = sorted(set([r['beta'] for r in results]))   # Unique betas

    # Create 2D arrays for heatmaps
    total_cov_matrix = np.full((len(beta_grid), len(alpha_grid)), np.nan)
    recall_matrix = np.full((len(beta_grid), len(alpha_grid)), np.nan)

    for r in results:
        alpha = r['alpha']
        beta = r['beta']
        temp = r['temp']
        total_cov = r['total_cov']
        avg_cov_per_group = r['avg_cov_per_group']
        i = beta_grid.index(beta)
        j = alpha_grid.index(alpha)
        total_cov_matrix[i, j] = total_cov
        recall_matrix[i, j] = avg_cov_per_group

    # Function to create annot matrix with bold max
    def create_annot_matrix(matrix):
        max_val = np.nanmax(matrix)
        annot = []
        for row in matrix:
            annot_row = []
            for val in row:
                if np.isnan(val):
                    annot_row.append("")
                elif val == max_val:
                    annot_row.append(r'$\mathbf{' + f'{val:.3f}' + '}$')
                else:
                    annot_row.append(f'{val:.3f}')
            annot.append(annot_row)
        return annot

    # Plot total coverage heatmap
    plt.figure(figsize=(8, 6))
    annot_total = create_annot_matrix(total_cov_matrix)
    sns.heatmap(total_cov_matrix, xticklabels=[f'{a:.1f}' for a in alpha_grid], 
                yticklabels=[f'{b:.1f}' for b in beta_grid], annot=annot_total, fmt='', cmap='Greens', vmin=0, vmax=1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Total Coverage Heatmap')
    plt.savefig(f'{save_dir}/heatmap_total_coverage.svg')  # Saves as SVG image
    plt.close()

    # Plot average recall per group heatmap
    plt.figure(figsize=(8, 6))
    annot_recall = create_annot_matrix(recall_matrix)
    sns.heatmap(recall_matrix, xticklabels=[f'{a:.1f}' for a in alpha_grid], 
                yticklabels=[f'{b:.1f}' for b in beta_grid], annot=annot_recall, fmt='', cmap='Greens', vmin=0, vmax=1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Average Coverage per Group Heatmap')
    plt.savefig(f'{save_dir}/heatmap_avg_coverage_per_group.svg')  # Saves as SVG image
    plt.close()

    print(f"Heatmaps and results saved to {save_dir}/")

def create_coverage_vs_size_plot(results, save_dir="./figures", relative_divisor=None):
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['total_cov'])  # Sort by total coverage
    # Save results to CSV using pandas for simplicity
    df = pd.DataFrame(results)
    df.to_csv(f'{save_dir}/coverage_vs_size_results.csv', index=False)
    create_coverage_vs_total_size_plot(results, save_dir=save_dir, relative_divisor=relative_divisor)
    create_coverage_vs_avg_group_size_plot(results, save_dir=save_dir, relative_divisor=relative_divisor)


def create_coverage_vs_total_size_plot(results, save_dir="./figures", filename="coverage_vs_total_size.svg", relative_divisor=None):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. candidate size.
    Saves the plot as an SVG file.

    Args:
        results: List of dicts with keys 'candidate_size', 'total_cov', 'avg_cov_per_group', 'pos_density', 'avg_group_density'
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        relative_divisor: Optional divisor for relative candidate size on second x-axis
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['candidate_size'])  # Sort by candidate size
    # Extract data
    candidate_sizes = [r['candidate_size'] for r in results]  # Total candidate size
    total_covs = [r['total_cov'] for r in results]
    avg_cov_per_groups = [r['avg_cov_per_group'] for r in results]
    pos_densities = [r['pos_density'] for r in results]
    avg_group_densities = [r['avg_group_density'] for r in results]
    # avg_group_counts = [r['avg_group_count'] for r in results]
    
    # Create the plot with dual y-axes

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis for coverage metrics
    ax1.plot(candidate_sizes, total_covs, label='Micro Coverage', marker='o', linestyle='-', color='darkgreen')
    ax1.plot(candidate_sizes, avg_cov_per_groups, label='Macro Coverage', marker='s', linestyle='--', color='seagreen')
    ax1.set_xlabel('Absolute Candidate Size')
    ax1.set_ylabel('Coverage')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(candidate_sizes, pos_densities, label='Micro Density', marker='^', linestyle=':', color='blue')
    ax2.plot(candidate_sizes, avg_group_densities, label='Macro Density', marker='v', linestyle='-.', color='dodgerblue')
    ax2.set_ylabel('Density')
    ax2.tick_params(axis='y')
    # ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Coverage and Density vs. Absolute Candidate Size')
    
    if relative_divisor is not None:
        ax3 = ax1.twiny()
        ax3.set_xlabel('Relative Candidate Size')
        ticks = ax1.get_xticks()
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([f'{tick / relative_divisor:.1f}' for tick in ticks])
        ax3.set_xlim(ax1.get_xlim())
        ax3.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_label_position('bottom')
        ax3.spines['bottom'].set_position(('outward', 40))
        ax1.spines['bottom'].set_position(('outward', 0))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot and results saved to {save_dir}/{filename}")

def create_coverage_vs_avg_group_size_plot(results, save_dir="./figures", filename="coverage_vs_avg_group_size.svg", relative_divisor=None):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. avg candidate size per group.
    Saves the plot as an SVG file.

    Args:
        results: List of dicts with keys 'candidate_size', 'total_cov', 'avg_cov_per_group', 'pos_density', 'avg_group_density', 'avg_group_count'
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        relative_divisor: Optional divisor for relative candidate size on second x-axis
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['avg_group_count'])  # Sort by avg candidate count per group
    # Extract data
    avg_group_counts = [r['avg_group_count'] for r in results]  # Avg candidate size per group
    total_covs = [r['total_cov'] for r in results]
    avg_cov_per_groups = [r['avg_cov_per_group'] for r in results]
    pos_densities = [r['pos_density'] for r in results]
    avg_group_densities = [r['avg_group_density'] for r in results]
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis for coverage metrics
    ax1.plot(avg_group_counts, total_covs, label='Micro Coverage', marker='o', linestyle='-', color='darkgreen')
    ax1.plot(avg_group_counts, avg_cov_per_groups, label='Macro Coverage', marker='s', linestyle='--', color='seagreen')
    ax1.set_xlabel('Macro Candidate Size')
    ax1.set_ylabel('Coverage')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(avg_group_counts, pos_densities, label='Micro Density', marker='^', linestyle=':', color='blue')
    ax2.plot(avg_group_counts, avg_group_densities, label='Macro Density', marker='v', linestyle='-.', color='dodgerblue')
    ax2.set_ylabel('Density', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Coverage and Density vs. Macro Candidate Size')
    
    if relative_divisor is not None:
        ax3 = ax1.twiny()
        ax3.set_xlabel('Relative Candidate Size')
        ticks = ax1.get_xticks()
        ax3.set_xticks(ticks)
        ax3.set_xticklabels([f'{tick / relative_divisor:.1f}' for tick in ticks])
        ax3.set_xlim(ax1.get_xlim())
        ax3.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_label_position('bottom')
        ax3.spines['bottom'].set_position(('outward', 40))
        ax1.spines['bottom'].set_position(('outward', 0))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot and results saved to {save_dir}/{filename}")

def call_plot_from_csv(csv_path, plot_func, **kwargs):
    """
    Reads a CSV file and calls the specified plotting function with the data.
    
    Args:
        csv_path: Path to the CSV file containing results.
        plot_func: The plotting function to call (e.g., create_coverage_vs_total_size_plot).
        save_dir: Directory to save the plot (default: "./figures").
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    df = pd.read_csv(csv_path)
    results = df.to_dict('records')
    plot_func(results, **kwargs)
p = "./logs/Final/Fb15k237/pathe2Phases/RelationPTailBce/version_0/figures/coverage_vs_size_results.csv"

def create_performance_vs_size_plot(results, save_dir="./figures", filename="performance_vs_size.svg"):
    """
    Create a line plot showing ranking metrics (Recall@10, MAP, NDCG) vs. candidate size.
    Saves the plot as an SVG file.

    Args:
        results: List of dicts with keys 'candidate_size', 'recall_10', 'map', 'ndcg'
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(f'{save_dir}/performance_vs_size_results.csv', index=False)
    
    results.sort(key=lambda x: x['candidate_size'])
    
    candidate_sizes = [r['candidate_size'] for r in results]
    recall_10 = [r['test_link_recall@10_perGroup'] for r in results]
    map_scores = [r['test_link_map'] for r in results]
    ndcg_scores = [r['test_link_ndcg'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(candidate_sizes, recall_10, label='Recall@10', marker='o', linestyle='-')
    ax.plot(candidate_sizes, map_scores, label='MAP', marker='s', linestyle='--')
    ax.plot(candidate_sizes, ndcg_scores, label='NDCG', marker='^', linestyle=':')
    
    ax.set_xlabel('Total Candidate Size')
    ax.set_ylabel('Metric Score')
    ax.set_title('Model Performance vs. Candidate Set Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Performance vs. size plot saved to {save_dir}/{filename}")


def create_relation_coverage_bar_chart(candidates, gold_triples, relation_maps, save_dir="./figures", filename="bar_relation_coverage.svg", num_bins=10):
    """
    Create a bar chart showing average coverage per quantile bin for relation types.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        gold_triples: (N, 3) tensor of gold triples
        relation_maps: RelationMaps object
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        num_bins: Number of quantile bins
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Count candidates per relation
    rel_counts = defaultdict(int)
    for cand in candidates:
        rel = cand[1].item()
        rel_counts[rel] += 1
    
    # Get unique relations from gold triples
    unique_rels = torch.unique(gold_triples[:, 1]).tolist()
    
    coverage_per_rel = {}
    freq_per_rel = {}
    for rel in unique_rels:
        gold_mask = gold_triples[:, 1] == rel
        gold_subset = gold_triples[gold_mask]
        cand_mask = candidates[:, 1] == rel
        cand_subset = candidates[cand_mask]
        
        if gold_subset.size(0) == 0:
            cov = 0.0
        else:
            covered = len(set(tuple(row.tolist()) for row in gold_subset) & set(tuple(row.tolist()) for row in cand_subset))
            cov = covered / gold_subset.size(0)
        coverage_per_rel[rel] = cov
        freq_per_rel[rel] = gold_subset.size(0)
    
    # Get coverages, counts, and freqs
    covs = []
    counts = []
    freqs = []
    for rel in unique_rels:
        covs.append(coverage_per_rel[rel])
        counts.append(rel_counts.get(rel, 0))
        freqs.append(freq_per_rel[rel])
    
    # Sort by coverage
    sorted_indices = np.argsort(freqs)
    covs_sorted = [covs[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]
    freqs_sorted = [freqs[i] for i in sorted_indices]
    
    # Divide into quantile bins of equal size
    cov_bins = np.array_split(covs_sorted, num_bins)
    count_bins = np.array_split(counts_sorted, num_bins)
    freq_bins = np.array_split(freqs_sorted, num_bins)
    avg_covs = []
    avg_counts = []
    avg_freqs = []
    bin_labels = []
    for i, (bin_covs, bin_counts, bin_freqs) in enumerate(zip(cov_bins, count_bins, freq_bins)):
        avg_cov = np.mean(bin_covs) if len(bin_covs) > 0 else 0.0
        avg_count = np.mean(bin_counts) if len(bin_counts) > 0 else 0.0
        avg_freq = np.mean(bin_freqs) if len(bin_freqs) > 0 else 0.0
        avg_covs.append(avg_cov)
        avg_counts.append(avg_count)
        avg_freqs.append(avg_freq)
        bin_labels.append(f"Q{i+1} ({len(bin_covs)} relations)")
    
    # Create bar chart with triple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(bin_labels))
    width = 0.25  # Width of bars
    
    # Primary axis for coverage
    bars1 = ax1.bar(x - width, avg_covs, width, label='Avg. Coverage per Relation', color='seagreen')
    ax1.set_xlabel('Frequency Quantile Bin')
    ax1.set_ylabel('Avg. Coverage per Relation', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)  # Coverage is between 0 and 1
    
    # Secondary axis for candidate counts
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, avg_counts, width, label='Avg. Candidates per Relation', color='blue')
    ax2.set_ylabel('Avg. Candidates per Relation', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Tertiary axis for frequencies (offset to the right)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))  # Offset the spine
    bars3 = ax3.bar(x + width, avg_freqs, width, label='Avg. Relation Frequency', color='orange')
    ax3.set_ylabel('Avg. Relation Frequency', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    plt.title('Average Coverage, Candidates, and Frequency per Relation Frequency Quantile')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_candidates_per_head_by_degree_chart(candidates, context_triple_store, gold_triples, save_dir="./figures", filename="candidates_per_head_by_degree.svg", degree_bins=25):
    """
    Create a bar chart showing average number of candidates per head and average coverage per head, grouped by entity degree bins.
    Uses quantile-based binning for better handling of skewed distributions.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        context_triple_store: List of triples for degree computation
        gold_triples: (N, 3) tensor of gold triples for coverage calculation
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        degree_bins: Number of quantile-based bins for degree grouping (e.g., 10 for deciles)
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Compute entity degrees from relational context
    entity_degrees = {}
    for context in context_triple_store:
        node_id = context[0].item()
        entity_degrees[node_id] = entity_degrees.get(node_id, 0) + 1
    
    # Count candidates per head
    head_counts = defaultdict(int)
    for cand in candidates:
        head = cand[0].item()
        head_counts[head] += 1
    
    # Compute coverage per head
    head_coverage = {}
    head_freq = {}
    for head in torch.unique(gold_triples[:, 0]):
        head = head.item()
        gold_mask = gold_triples[:, 0] == head
        gold_subset = gold_triples[gold_mask]
        cand_mask = candidates[:, 0] == head
        cand_subset = candidates[cand_mask]
        
        if gold_subset.size(0) == 0:
            cov = 0.0
        else:
            covered = len(set(tuple(row.tolist()) for row in gold_subset) & set(tuple(row.tolist()) for row in cand_subset))
            cov = covered / gold_subset.size(0)
        head_coverage[head] = cov
        head_freq[head] = gold_subset.size(0)
    
    # Group by degree
    degrees = [entity_degrees.get(h, 0) for h in head_counts.keys()]
    counts = list(head_counts.values())
    coverages = [head_coverage.get(h, 0.0) for h in head_counts.keys()]
    freqs = [head_freq.get(h, 0) for h in head_counts.keys()]
    
    # Bin degrees using quantiles
    if degrees:
        # Compute quantile edges (e.g., deciles for degree_bins=10)
        quantiles = np.quantile(degrees, np.linspace(0, 1, degree_bins + 1))
        # Ensure unique quantiles (handle ties)
        quantiles = np.unique(quantiles)
        if len(quantiles) < degree_bins + 1:
            # If not enough unique quantiles, fall back to linear bins
            max_deg = max(degrees)
            quantiles = np.linspace(0, max_deg, degree_bins + 1)
        
        bin_indices = np.digitize(degrees, quantiles) - 1
        bin_indices = np.clip(bin_indices, 0, len(quantiles) - 2)
        
        avg_counts_per_bin = []
        avg_coverages_per_bin = []
        avg_freqs_per_bin = []
        bin_labels = []
        for i in range(len(quantiles) - 1):
            mask = bin_indices == i
            bin_degrees = [degrees[j] for j in range(len(degrees)) if mask[j]]
            if bin_degrees:
                min_deg = min(bin_degrees)
                max_deg = max(bin_degrees)
                avg_count = np.mean([counts[j] for j in range(len(counts)) if mask[j]])
                avg_cov = np.mean([coverages[j] for j in range(len(coverages)) if mask[j]])
                avg_freq = np.mean([freqs[j] for j in range(len(freqs)) if mask[j]])
            else:
                min_deg = 0
                max_deg = 0
                avg_count = 0
                avg_cov = 0.0
                avg_freq = 0.0
            avg_counts_per_bin.append(avg_count)
            avg_coverages_per_bin.append(avg_cov)
            avg_freqs_per_bin.append(avg_freq)
            bin_labels.append(f"Q{i+1} ({min_deg}-{max_deg})")
    else:
        avg_counts_per_bin = [0] * degree_bins
        avg_coverages_per_bin = [0.0] * degree_bins
        bin_labels = [f"Q{i+1} (0-0)" for i in range(degree_bins)]
    
    # Create bar chart with triple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(bin_labels))
    width = 0.25  # Width of bars
    
    # Primary axis for coverage
    bars1 = ax1.bar(x - width, avg_coverages_per_bin, width, label='Avg. Coverage per Head', color='seagreen')
    ax1.set_xlabel('Entity Degree Quantile Bin')
    ax1.set_ylabel('Avg. Coverage per Head', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)  # Coverage is between 0 and 1
    
    # Secondary axis for candidate counts
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, avg_counts_per_bin, width, label='Avg. Candidates per Head', color='blue')
    ax2.set_ylabel('Avg. Candidates per Head', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Tertiary axis for frequencies (offset to the right)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))  # Offset the spine
    bars3 = ax3.bar(x + width, avg_freqs_per_bin, width, label='Avg. Head Frequency', color='orange')
    ax3.set_ylabel('Avg. Head Frequency', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    plt.title('Average Candidates, Coverage, and Frequency per Head by Degree (Quantile Bins)')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_entity_coverage_bar_chart(candidates, gold_triples, context_triple_store, save_dir="./figures", filename="bar_entity_coverage.svg", num_bins=25):
    """
    Create a bar chart showing average coverage per quantile bin for head entities.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        gold_triples: (N, 3) tensor of gold triples
        context_triple_store: List of triples for degree computation
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        num_bins: Number of quantile bins
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute entity degrees from context_triple_store
    entity_degrees = {}
    for context in context_triple_store:
        node_id = context[0].item()
        entity_degrees[node_id] = entity_degrees.get(node_id, 0) + 1
    
    # Count candidates per head
    head_counts = defaultdict(int)
    for cand in candidates:
        head = cand[0].item()
        head_counts[head] += 1
    
    # Get unique heads from gold triples
    unique_heads = torch.unique(gold_triples[:, 0]).tolist()
    
    coverage_per_head = {}
    freq_per_head = {}
    for head in unique_heads:
        gold_mask = gold_triples[:, 0] == head
        gold_subset = gold_triples[gold_mask]
        cand_mask = candidates[:, 0] == head
        cand_subset = candidates[cand_mask]
        
        if gold_subset.size(0) == 0:
            cov = 0.0
        else:
            covered = len(set(tuple(row.tolist()) for row in gold_subset) & set(tuple(row.tolist()) for row in cand_subset))
            cov = covered / gold_subset.size(0)
        coverage_per_head[head] = cov
        freq_per_head[head] = gold_subset.size(0)
    
    # Get coverages, counts, and freqs
    covs = []
    counts = []
    freqs = []
    for head in unique_heads:
        covs.append(coverage_per_head[head])
        counts.append(head_counts.get(head, 0))
        freqs.append(freq_per_head[head])
    
    # Sort by coverage
    sorted_indices = np.argsort(freqs)
    covs_sorted = [covs[i] for i in sorted_indices]
    counts_sorted = [counts[i] for i in sorted_indices]
    freqs_sorted = [freqs[i] for i in sorted_indices]
    
    # Divide into quantile bins of equal size
    cov_bins = np.array_split(covs_sorted, num_bins)
    count_bins = np.array_split(counts_sorted, num_bins)
    freq_bins = np.array_split(freqs_sorted, num_bins)
    avg_covs = []
    avg_counts = []
    avg_freqs = []
    bin_labels = []
    for i, (bin_covs, bin_counts, bin_freqs) in enumerate(zip(cov_bins, count_bins, freq_bins)):
        avg_cov = np.mean(bin_covs) if len(bin_covs) > 0 else 0.0
        avg_count = np.mean(bin_counts) if len(bin_counts) > 0 else 0.0
        avg_freq = np.mean(bin_freqs) if len(bin_freqs) > 0 else 0.0
        avg_covs.append(avg_cov)
        avg_counts.append(avg_count)
        avg_freqs.append(avg_freq)
        bin_labels.append(f"Q{i+1} ({len(bin_covs)} heads)")
    
    # Create bar chart with triple y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(bin_labels))
    width = 0.25  # Width of bars
    
    # Primary axis for coverage
    bars1 = ax1.bar(x - width, avg_covs, width, label='Avg. Coverage per Head', color='seagreen')
    ax1.set_xlabel('Frequency Quantile Bin')
    ax1.set_ylabel('Avg. Coverage per Head', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)  # Coverage is between 0 and 1
    
    # Secondary axis for candidate counts
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x, avg_counts, width, label='Avg. Candidates per Head', color='blue')
    ax2.set_ylabel('Avg. Candidates per Head', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Tertiary axis for frequencies (offset to the right)
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))  # Offset the spine
    bars3 = ax3.bar(x + width, avg_freqs, width, label='Avg. Head Frequency', color='orange')
    ax3.set_ylabel('Avg. Head Frequency', color='orange')
    ax3.tick_params(axis='y', labelcolor='orange')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')
    
    plt.title('Average Coverage, Candidates, and Frequency per Head Entity Frequency Quantile')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_test_set_statistics_figure(context_triple_store, train_triples, save_dir="./figures", filename="test_set_statistics.svg"):
    """
    Create a figure showing statistics of the test set graph, including degree distributions, relation frequencies, and summary stats.
    Saves the plot as an SVG file.

    Args:
        context_triple_store: List of triples for degree computation
        train_triples: (N, 3) tensor of training triples
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute entity degrees from context_triple_store
    entity_degrees_in = defaultdict(int)
    entity_degrees_out = defaultdict(int)
    for triple in context_triple_store:
        head = triple[0].item()
        tail = triple[2].item()
        entity_degrees_out[head] += 1
        entity_degrees_in[tail] += 1
    
    # Use train_triples
    triples = train_triples
    num_triples = triples.size(0)
    unique_entities = torch.unique(triples[:, [0, 2]]).tolist()
    num_entities = len(unique_entities)
    unique_relations = torch.unique(triples[:, 1]).tolist()
    num_relations = len(unique_relations)
    
    # Relation frequencies
    relation_counts = defaultdict(int)
    for triple in triples:
        relation_counts[triple[1].item()] += 1
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # In-degree distribution
    in_degrees = list(entity_degrees_in.values())
    axes[0, 0].hist(in_degrees, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('In-degree Distribution')
    axes[0, 0].set_xlabel('Degree')
    axes[0, 0].set_ylabel('Frequency')
    
    # Out-degree distribution
    out_degrees = list(entity_degrees_out.values())
    axes[0, 1].hist(out_degrees, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Out-degree Distribution')
    axes[0, 1].set_xlabel('Degree')
    axes[0, 1].set_ylabel('Frequency')
    
    # Relation frequency distribution
    rel_freqs = list(relation_counts.values())
    axes[1, 0].hist(rel_freqs, bins=50, alpha=0.7, color='red')
    axes[1, 0].set_title('Relation Frequency Distribution')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].set_ylabel('Number of Relations')
    
    # Summary statistics
    summary = f"Entities: {num_entities}\nRelations: {num_relations}\nTriples: {num_triples}"
    axes[1, 1].text(0.1, 0.5, summary, fontsize=12, verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Summary Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Figure saved to {save_dir}/{filename}")

def create_entity_occurrence_figure(triples, save_dir="./figures", filename="entity_occurrence.svg", max_samples=1000000):
    """
    Create histograms showing the distribution of entity occurrence counts in knowledge graph triples.
    Generates two subplots: one for tail counts per head (out-degree distribution), and one for head counts per tail (in-degree distribution).
    This provides statistics on the connectivity patterns of entities.

    Args:
        triples: (N, 3) tensor of triples (head, relation, tail)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        max_samples: Max samples for histogram to avoid OOM (samples if data > max_samples)
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to CPU if needed
    triples = triples.cpu()
    
    # Get all unique entities
    all_heads = torch.unique(triples[:, 0]).tolist()
    all_tails = torch.unique(triples[:, 2]).tolist()
    
    # Compute tail counts per head
    tail_counts_per_head = []
    for head in all_heads:
        head_triples = triples[triples[:, 0] == head]
        tail_counts = torch.bincount(head_triples[:, 2], minlength=len(all_tails))
        tail_counts_per_head.extend(tail_counts.tolist())
    
    # Compute head counts per tail
    head_counts_per_tail = []
    for tail in all_tails:
        tail_triples = triples[triples[:, 2] == tail]
        head_counts = torch.bincount(tail_triples[:, 0], minlength=len(all_heads))
        head_counts_per_tail.extend(head_counts.tolist())
    
    # Sample if too large
    if len(tail_counts_per_head) > max_samples:
        import random
        tail_counts_per_head = random.sample(tail_counts_per_head, max_samples)
        print(f"Sampled {max_samples} tail counts per head for histogram to avoid OOM.")
    
    if len(head_counts_per_tail) > max_samples:
        import random
        head_counts_per_tail = random.sample(head_counts_per_tail, max_samples)
        print(f"Sampled {max_samples} head counts per tail for histogram to avoid OOM.")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First subplot: tail counts per head
    if tail_counts_per_head:
        min_count1 = min(tail_counts_per_head)
        max_count1 = max(tail_counts_per_head)
        bins1 = np.arange(min_count1 - 0.5, max_count1 + 1.5, 1)
        ax1.hist(tail_counts_per_head, bins=bins1, alpha=0.7, color='purple', edgecolor='black')
        ax1.set_xlabel('Number of Tails per Head')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Tail Counts per Head')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        # Set xticks with reasonable spacing
        step1 = max(1, (max_count1 - min_count1) // 10)
        ax1.set_xticks(np.arange(min_count1, max_count1 + 1, step1))
    
    # Second subplot: head counts per tail
    if head_counts_per_tail:
        min_count2 = min(head_counts_per_tail)
        max_count2 = max(head_counts_per_tail)
        bins2 = np.arange(min_count2 - 0.5, max_count2 + 1.5, 1)
        ax2.hist(head_counts_per_tail, bins=bins2, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Number of Heads per Tail')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Head Counts per Tail')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        # Set xticks with reasonable spacing
        step2 = max(1, (max_count2 - min_count2) // 10)
        ax2.set_xticks(np.arange(min_count2, max_count2 + 1, step2))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Entity occurrence figure saved to {save_dir}/{filename}")

def create_relation_occurrence_figure(triples, save_dir="./figures", filename="relation_occurrence.svg", max_samples=1000000):
    """
    Create histograms showing the distribution of relation occurrence counts in knowledge graph triples.
    Generates two subplots: one for relation counts per head, and one for relation counts per tail.
    This provides statistics on the relational connectivity patterns of entities.

    Args:
        triples: (N, 3) tensor of triples (head, relation, tail)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        max_samples: Max samples for histogram to avoid OOM (samples if data > max_samples)
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Move to CPU if needed
    triples = triples.cpu()
    
    # Get all unique entities and relations
    all_heads = torch.unique(triples[:, 0]).tolist()
    all_tails = torch.unique(triples[:, 2]).tolist()
    all_relations = torch.unique(triples[:, 1]).tolist()
    
    # Compute relation counts per head
    rel_counts_per_head = []
    for head in all_heads:
        head_triples = triples[triples[:, 0] == head]
        rel_counts = torch.bincount(head_triples[:, 1], minlength=len(all_relations))
        rel_counts_per_head.extend(rel_counts.tolist())
    
    # Compute relation counts per tail
    rel_counts_per_tail = []
    for tail in all_tails:
        tail_triples = triples[triples[:, 2] == tail]
        rel_counts = torch.bincount(tail_triples[:, 1], minlength=len(all_relations))
        rel_counts_per_tail.extend(rel_counts.tolist())
    
    # Sample if too large
    if len(rel_counts_per_head) > max_samples:
        import random
        rel_counts_per_head = random.sample(rel_counts_per_head, max_samples)
        print(f"Sampled {max_samples} relation counts per head for histogram to avoid OOM.")
    
    if len(rel_counts_per_tail) > max_samples:
        import random
        rel_counts_per_tail = random.sample(rel_counts_per_tail, max_samples)
        print(f"Sampled {max_samples} relation counts per tail for histogram to avoid OOM.")
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # First subplot: relation counts per head
    if rel_counts_per_head:
        min_count1 = min(rel_counts_per_head)
        max_count1 = max(rel_counts_per_head)
        bins1 = np.arange(min_count1 - 0.5, max_count1 + 1.5, 1)
        ax1.hist(rel_counts_per_head, bins=bins1, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel('Number of Relations per Head')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Relation Counts per Head')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        # Set xticks with reasonable spacing
        step1 = max(1, (max_count1 - min_count1) // 10)
        ax1.set_xticks(np.arange(min_count1, max_count1 + 1, step1))
    
    # Second subplot: relation counts per tail
    if rel_counts_per_tail:
        min_count2 = min(rel_counts_per_tail)
        max_count2 = max(rel_counts_per_tail)
        bins2 = np.arange(min_count2 - 0.5, max_count2 + 1.5, 1)
        ax2.hist(rel_counts_per_tail, bins=bins2, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Number of Relations per Tail')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Relation Counts per Tail')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        # Set xticks with reasonable spacing
        step2 = max(1, (max_count2 - min_count2) // 10)
        ax2.set_xticks(np.arange(min_count2, max_count2 + 1, step2))
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Relation occurrence figure saved to {save_dir}/{filename}")


def plot_relation_count_dispersion(triples, save_dir="./figures", filename="relation_dispersion.svg"):
    """
    Analyzes if relation type counts per relation follow a Poisson distribution by plotting variance vs. mean.
    Creates two subplots: one for relation type counts across heads, and one for relation type counts across tails.
    For each relation type, computes the mean and variance of its occurrence counts across all heads/tails (including 0 for heads/tails without that relation).

    Args:
        triples (torch.Tensor): (N, 3) tensor of (head, relation, tail).
        save_dir (str): Directory to save the figure.
        filename (str): Name of the output SVG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data structures
    rel_to_head_counts = defaultdict(lambda: defaultdict(int))
    rel_to_tail_counts = defaultdict(lambda: defaultdict(int))
    all_relations = set()
    all_heads = set()
    all_tails = set()

    for h, r, t in triples:
        h, r, t = h.item(), r.item(), t.item()
        rel_to_head_counts[r][h] += 1
        rel_to_tail_counts[r][t] += 1
        all_relations.add(r)
        all_heads.add(h)
        all_tails.add(t)

    # Compute for heads
    variances_h, means_h, sample_sizes_h = [], [], []
    variances_nonzero_h, means_nonzero_h, sample_sizes_nonzero_h = [], [], []

    # Compute for tails
    variances_t, means_t, sample_sizes_t = [], [], []
    variances_nonzero_t, means_nonzero_t, sample_sizes_nonzero_t = [], [], []

    for rel in all_relations:
        # Heads
        counts_h = [rel_to_head_counts[rel].get(h, 0) for h in all_heads]
        if len(counts_h) > 1:
            means_h.append(np.mean(counts_h))
            variances_h.append(np.var(counts_h))
            sample_sizes_h.append(len(counts_h))
            counts_nonzero_h = [c for c in counts_h if c > 0]
            if len(counts_nonzero_h) > 1:
                means_nonzero_h.append(np.mean(counts_nonzero_h))
                variances_nonzero_h.append(np.var(counts_nonzero_h))
                sample_sizes_nonzero_h.append(len(counts_nonzero_h))

        # Tails
        counts_t = [rel_to_tail_counts[rel].get(t, 0) for t in all_tails]
        if len(counts_t) > 1:
            means_t.append(np.mean(counts_t))
            variances_t.append(np.var(counts_t))
            sample_sizes_t.append(len(counts_t))
            counts_nonzero_t = [c for c in counts_t if c > 0]
            if len(counts_nonzero_t) > 1:
                means_nonzero_t.append(np.mean(counts_nonzero_t))
                variances_nonzero_t.append(np.var(counts_nonzero_t))
                sample_sizes_nonzero_t.append(len(counts_nonzero_t))

    if not means_h or not means_t:
        print("No data to plot for dispersion.")
        return

    # Compute pooled dispersion indices
    if variances_h and sample_sizes_h:
        numerator_h = np.sum([var * (n-1) for var, n in zip(variances_h, sample_sizes_h)])
        denominator_h = np.sum([mu * (n-1) for mu, n in zip(means_h, sample_sizes_h)])
        pooled_dispersion_h = numerator_h / denominator_h if denominator_h > 0 else float('nan')
    else:
        pooled_dispersion_h = float('nan')

    if variances_nonzero_h and sample_sizes_nonzero_h:
        numerator_nonzero_h = np.sum([var * (n-1) for var, n in zip(variances_nonzero_h, sample_sizes_nonzero_h)])
        denominator_nonzero_h = np.sum([mu * (n-1) for mu, n in zip(means_nonzero_h, sample_sizes_nonzero_h)])
        pooled_dispersion_nonzero_h = numerator_nonzero_h / denominator_nonzero_h if denominator_nonzero_h > 0 else float('nan')
    else:
        pooled_dispersion_nonzero_h = float('nan')

    if variances_t and sample_sizes_t:
        numerator_t = np.sum([var * (n-1) for var, n in zip(variances_t, sample_sizes_t)])
        denominator_t = np.sum([mu * (n-1) for mu, n in zip(means_t, sample_sizes_t)])
        pooled_dispersion_t = numerator_t / denominator_t if denominator_t > 0 else float('nan')
    else:
        pooled_dispersion_t = float('nan')

    if variances_nonzero_t and sample_sizes_nonzero_t:
        numerator_nonzero_t = np.sum([var * (n-1) for var, n in zip(variances_nonzero_t, sample_sizes_nonzero_t)])
        denominator_nonzero_t = np.sum([mu * (n-1) for mu, n in zip(means_nonzero_t, sample_sizes_nonzero_t)])
        pooled_dispersion_nonzero_t = numerator_nonzero_t / denominator_nonzero_t if denominator_nonzero_t > 0 else float('nan')
    else:
        pooled_dispersion_nonzero_t = float('nan')

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # First subplot: for heads
    ax1.scatter(means_h, variances_h, alpha=0.5, color='red', label='Including 0s')
    ax1.scatter(means_nonzero_h, variances_nonzero_h, alpha=0.5, color='blue', label='Excluding 0s')
    max_val_h = max(max(means_h + means_nonzero_h), max(variances_h + variances_nonzero_h))
    ax1.plot([0, max_val_h], [0, max_val_h], 'k--', label='y=x (Poisson ideal)')
    ax1.set_xlabel('Mean Relation Type Count per Head')
    ax1.set_ylabel('Variance of Relation Type Count per Head')
    ax1.set_title('Dispersion of Relation Type Counts Across Heads')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # ax1.axis('equal')
    # ax1.set_xlim(left=0)
    # ax1.set_ylim(bottom=0)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    textstr_h = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_h:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_h:.2f}'
    ax1.text(0.95, 0.05, textstr_h, transform=ax1.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Second subplot: for tails
    ax2.scatter(means_t, variances_t, alpha=0.5, color='red', label='Including 0s')
    ax2.scatter(means_nonzero_t, variances_nonzero_t, alpha=0.5, color='blue', label='Excluding 0s')
    max_val_t = max(max(means_t + means_nonzero_t), max(variances_t + variances_nonzero_t))
    ax2.plot([0, max_val_t], [0, max_val_t], 'k--', label='y=x (Poisson ideal)')
    ax2.set_xlabel('Mean Relation Type Count per Tail')
    ax2.set_ylabel('Variance of Relation Type Count per Tail')
    ax2.set_title('Dispersion of Relation Type Counts Across Tails')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    # ax2.axis('equal')
    # ax2.set_xlim(left=0)
    # ax2.set_ylim(bottom=0)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    textstr_t = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_t:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_t:.2f}'
    ax2.text(0.95, 0.05, textstr_t, transform=ax2.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Dispersion plot saved to {os.path.join(save_dir, filename)}")
    print(f"Heads - Pooled Dispersion Index (Including 0s): {pooled_dispersion_h:.2f}")
    print(f"Heads - Pooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_h:.2f}")
    print(f"Tails - Pooled Dispersion Index (Including 0s): {pooled_dispersion_t:.2f}")
    print(f"Tails - Pooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_t:.2f}")

def plot_entity_pair_multiplicity_dispersion(triples, save_dir="./figures", filename="entity_pair_dispersion.svg"):
    """
    Analyzes the dispersion of Entity Pair Multiplicity (h, t).
    Counts how many distinct relation types connect a specific pair of entities.
    y_{h,t} = |{r \in R | (h, r, t) \in T}|
    
    Plots Variance vs Mean for:
    1. Grouped by Head: For each head h, statistics of y_{h,t} across all tails t.
    2. Grouped by Tail: For each tail t, statistics of y_{h,t} across all heads h.

    Args:
        triples (torch.Tensor): (N, 3) tensor of (head, relation, tail).
        save_dir (str): Directory to save the figure.
        filename (str): Name of the output SVG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data structures
    # head_to_tail_counts[h][t] = count of relations between h and t
    head_to_tail_counts = defaultdict(lambda: defaultdict(int))
    # tail_to_head_counts[t][h] = count of relations between h and t
    tail_to_head_counts = defaultdict(lambda: defaultdict(int))
    
    all_heads = set()
    all_tails = set()

    for h, r, t in triples:
        h, t = h.item(), t.item()
        head_to_tail_counts[h][t] += 1
        tail_to_head_counts[t][h] += 1
        all_heads.add(h)
        all_tails.add(t)

    num_heads = len(all_heads)
    num_tails = len(all_tails)
    print(f"Total unique heads: {num_heads}, tails: {num_tails}, intersection: {len(all_heads & all_tails)}, union: {len(all_heads | all_tails)}")

    # --- Compute Statistics Grouped by Head ---
    # For each head, we look at the distribution of counts across all tails
    means_h, variances_h, sample_sizes_h = [], [], []
    means_nonzero_h, variances_nonzero_h, sample_sizes_nonzero_h = [], [], []

    for h in all_heads:
        counts_nonzero = list(head_to_tail_counts[h].values())
        
        # Excluding 0s
        if len(counts_nonzero) > 1:
            means_nonzero_h.append(np.mean(counts_nonzero))
            variances_nonzero_h.append(np.var(counts_nonzero))
            sample_sizes_nonzero_h.append(len(counts_nonzero))
        
        # Including 0s (Sparse calculation)
        # Total population is all_tails
        sum_x = sum(counts_nonzero)
        sum_sq_x = sum(c**2 for c in counts_nonzero)
        
        mean_all = sum_x / num_tails
        mean_sq_all = sum_sq_x / num_tails
        var_all = mean_sq_all - (mean_all ** 2)
        
        means_h.append(mean_all)
        variances_h.append(var_all)
        sample_sizes_h.append(num_tails)

    # --- Compute Statistics Grouped by Tail ---
    # For each tail, we look at the distribution of counts across all heads
    means_t, variances_t, sample_sizes_t = [], [], []
    means_nonzero_t, variances_nonzero_t, sample_sizes_nonzero_t = [], [], []

    for t in all_tails:
        counts_nonzero = list(tail_to_head_counts[t].values())
        
        # Excluding 0s
        if len(counts_nonzero) > 1:
            means_nonzero_t.append(np.mean(counts_nonzero))
            variances_nonzero_t.append(np.var(counts_nonzero))
            sample_sizes_nonzero_t.append(len(counts_nonzero))
            
        # Including 0s (Sparse calculation)
        # Total population is all_heads
        sum_x = sum(counts_nonzero)
        sum_sq_x = sum(c**2 for c in counts_nonzero)
        
        mean_all = sum_x / num_heads
        mean_sq_all = sum_sq_x / num_heads
        var_all = mean_sq_all - (mean_all ** 2)
        
        means_t.append(mean_all)
        variances_t.append(var_all)
        sample_sizes_t.append(num_heads)

    # --- Compute Pooled Dispersion Indices ---
    def compute_pooled_dispersion(variances, means, sample_sizes):
        if variances and sample_sizes:
            numerator = np.sum([var * (n-1) for var, n in zip(variances, sample_sizes)])
            denominator = np.sum([mu * (n-1) for mu, n in zip(means, sample_sizes)])
            return numerator / denominator if denominator > 0 else float('nan')
        return float('nan')

    pooled_dispersion_h = compute_pooled_dispersion(variances_h, means_h, sample_sizes_h)
    pooled_dispersion_nonzero_h = compute_pooled_dispersion(variances_nonzero_h, means_nonzero_h, sample_sizes_nonzero_h)
    
    pooled_dispersion_t = compute_pooled_dispersion(variances_t, means_t, sample_sizes_t)
    pooled_dispersion_nonzero_t = compute_pooled_dispersion(variances_nonzero_t, means_nonzero_t, sample_sizes_nonzero_t)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Subplot 1: Grouped by Head
    if means_h:
        ax1.scatter(means_h, variances_h, alpha=0.5, color='red', label='Including 0s')
        max_val_h = max(max(means_h), max(variances_h)) if means_h else 1
        if means_nonzero_h:
             ax1.scatter(means_nonzero_h, variances_nonzero_h, alpha=0.5, color='blue', label='Excluding 0s')
             max_val_h = max(max_val_h, max(means_nonzero_h), max(variances_nonzero_h))
        
        ax1.plot([0, max_val_h], [0, max_val_h], 'k--', label='y=x (Poisson ideal)')
        ax1.set_xlabel('Mean Relations per Tail (for a Head)')
        ax1.set_ylabel('Variance of Relations per Tail (for a Head)')
        ax1.set_title('Dispersion of Entity Pair Multiplicity (Grouped by Head)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        textstr_h = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_h:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_h:.2f}'
        ax1.text(0.95, 0.05, textstr_h, transform=ax1.transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: Grouped by Tail
    if means_t:
        ax2.scatter(means_t, variances_t, alpha=0.5, color='red', label='Including 0s')
        max_val_t = max(max(means_t), max(variances_t)) if means_t else 1
        if means_nonzero_t:
            ax2.scatter(means_nonzero_t, variances_nonzero_t, alpha=0.5, color='blue', label='Excluding 0s')
            max_val_t = max(max_val_t, max(means_nonzero_t), max(variances_nonzero_t))

        ax2.plot([0, max_val_t], [0, max_val_t], 'k--', label='y=x (Poisson ideal)')
        ax2.set_xlabel('Mean Relations per Head (for a Tail)')
        ax2.set_ylabel('Variance of Relations per Head (for a Tail)')
        ax2.set_title('Dispersion of Entity Pair Multiplicity (Grouped by Tail)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        
        textstr_t = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_t:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_t:.2f}'
        ax2.text(0.95, 0.05, textstr_t, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Entity pair dispersion plot saved to {os.path.join(save_dir, filename)}")
    print(f"Grouped by Head - Pooled Dispersion Index (Including 0s): {pooled_dispersion_h:.2f}")
    print(f"Grouped by Head - Pooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_h:.2f}")
    print(f"Grouped by Tail - Pooled Dispersion Index (Including 0s): {pooled_dispersion_t:.2f}")
    print(f"Grouped by Tail - Pooled Dispersion Index (Excluding 0# filepath: /home/juliansampels/Masterarbeit/MasterThesis/PathE/pathe/figures.pys): {pooled_dispersion_nonzero_t:.2f}")



def plot_entity_degree_dispersion(triples, save_dir="./figures", filename="entity_dispersion.svg"):
    """
    Analyzes the dispersion of entity counts in knowledge graph triples.
    Creates two subplots: one for head counts per tail (in-degree dispersion), and one for tail counts per head (out-degree dispersion).
    For each entity, computes the mean and variance of its connection counts across the opposite entities (including 0 for missing connections).

    Args:
        triples (torch.Tensor): (N, 3) tensor of (head, relation, tail).
        save_dir (str): Directory to save the figure.
        filename (str): Name of the output SVG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Initialize data structures
    tail_to_head_counts = defaultdict(lambda: defaultdict(int))
    head_to_tail_counts = defaultdict(lambda: defaultdict(int))
    all_tails = set()
    all_heads = set()

    for h, r, t in triples:
        h, t = h.item(), t.item()
        tail_to_head_counts[t][h] += 1
        head_to_tail_counts[h][t] += 1
        all_tails.add(t)
        all_heads.add(h)

    # Compute for head counts per tail
    means_ht, variances_ht, sample_sizes_ht = [], [], []
    means_nonzero_ht, variances_nonzero_ht, sample_sizes_nonzero_ht = [], [], []

    # Compute for tail counts per head
    means_th, variances_th, sample_sizes_th = [], [], []
    means_nonzero_th, variances_nonzero_th, sample_sizes_nonzero_th = [], [], []

    for tail in all_tails:
        counts_ht = [tail_to_head_counts[tail].get(h, 0) for h in all_heads]
        if len(counts_ht) > 1:
            means_ht.append(np.mean(counts_ht))
            variances_ht.append(np.var(counts_ht))
            sample_sizes_ht.append(len(counts_ht))
            counts_nonzero_ht = [c for c in counts_ht if c > 0]
            if len(counts_nonzero_ht) > 1:
                means_nonzero_ht.append(np.mean(counts_nonzero_ht))
                variances_nonzero_ht.append(np.var(counts_nonzero_ht))
                sample_sizes_nonzero_ht.append(len(counts_nonzero_ht))

    for head in all_heads:
        counts_th = [head_to_tail_counts[head].get(t, 0) for t in all_tails]
        if len(counts_th) > 1:
            means_th.append(np.mean(counts_th))
            variances_th.append(np.var(counts_th))
            sample_sizes_th.append(len(counts_th))
            counts_nonzero_th = [c for c in counts_th if c > 0]
            if len(counts_nonzero_th) > 1:
                means_nonzero_th.append(np.mean(counts_nonzero_th))
                variances_nonzero_th.append(np.var(counts_nonzero_th))
                sample_sizes_nonzero_th.append(len(counts_nonzero_th))

    if not means_ht or not means_th:
        print("No data to plot for entity dispersion.")
        return

    # Compute pooled dispersion indices
    if variances_ht and sample_sizes_ht:
        numerator_ht = np.sum([var * (n-1) for var, n in zip(variances_ht, sample_sizes_ht)])
        denominator_ht = np.sum([mu * (n-1) for mu, n in zip(means_ht, sample_sizes_ht)])
        pooled_dispersion_ht = numerator_ht / denominator_ht if denominator_ht > 0 else float('nan')
    else:
        pooled_dispersion_ht = float('nan')

    if variances_nonzero_ht and sample_sizes_nonzero_ht:
        numerator_nonzero_ht = np.sum([var * (n-1) for var, n in zip(variances_nonzero_ht, sample_sizes_nonzero_ht)])
        denominator_nonzero_ht = np.sum([mu * (n-1) for mu, n in zip(means_nonzero_ht, sample_sizes_nonzero_ht)])
        pooled_dispersion_nonzero_ht = numerator_nonzero_ht / denominator_nonzero_ht if denominator_nonzero_ht > 0 else float('nan')
    else:
        pooled_dispersion_nonzero_ht = float('nan')

    if variances_th and sample_sizes_th:
        numerator_th = np.sum([var * (n-1) for var, n in zip(variances_th, sample_sizes_th)])
        denominator_th = np.sum([mu * (n-1) for mu, n in zip(means_th, sample_sizes_th)])
        pooled_dispersion_th = numerator_th / denominator_th if denominator_th > 0 else float('nan')
    else:
        pooled_dispersion_th = float('nan')

    if variances_nonzero_th and sample_sizes_nonzero_th:
        numerator_nonzero_th = np.sum([var * (n-1) for var, n in zip(variances_nonzero_th, sample_sizes_nonzero_th)])
        denominator_nonzero_th = np.sum([mu * (n-1) for mu, n in zip(means_nonzero_th, sample_sizes_nonzero_th)])
        pooled_dispersion_nonzero_th = numerator_nonzero_th / denominator_nonzero_th if denominator_nonzero_th > 0 else float('nan')
    else:
        pooled_dispersion_nonzero_th = float('nan')

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # First subplot: head counts per tail
    ax1.scatter(means_ht, variances_ht, alpha=0.5, color='red', label='Including 0s')
    ax1.scatter(means_nonzero_ht, variances_nonzero_ht, alpha=0.5, color='blue', label='Excluding 0s')
    max_val_ht = max(max(means_ht + means_nonzero_ht), max(variances_ht + variances_nonzero_ht))
    ax1.plot([0, max_val_ht], [0, max_val_ht], 'k--', label='y=x (Poisson ideal)')
    ax1.set_xlabel('Mean Head Count per Tail')
    ax1.set_ylabel('Variance of Head Count per Tail')
    ax1.set_title('Dispersion of Head Counts Across Tails')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    textstr_ht = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_ht:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_ht:.2f}'
    ax1.text(0.95, 0.05, textstr_ht, transform=ax1.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Second subplot: tail counts per head
    ax2.scatter(means_th, variances_th, alpha=0.5, color='red', label='Including 0s')
    ax2.scatter(means_nonzero_th, variances_nonzero_th, alpha=0.5, color='blue', label='Excluding 0s')
    max_val_th = max(max(means_th + means_nonzero_th), max(variances_th + variances_nonzero_th))
    ax2.plot([0, max_val_th], [0, max_val_th], 'k--', label='y=x (Poisson ideal)')
    ax2.set_xlabel('Mean Tail Count per Head')
    ax2.set_ylabel('Variance of Tail Count per Head')
    ax2.set_title('Dispersion of Tail Counts Across Heads')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    textstr_th = f'Pooled Dispersion Index (Including 0s): {pooled_dispersion_th:.2f}\nPooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_th:.2f}'
    ax2.text(0.95, 0.05, textstr_th, transform=ax2.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Entity dispersion plot saved to {os.path.join(save_dir, filename)}")
    print(f"Tails - Pooled Dispersion Index (Including 0s): {pooled_dispersion_ht:.2f}")
    print(f"Tails - Pooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_ht:.2f}")
    print(f"Heads - Pooled Dispersion Index (Including 0s): {pooled_dispersion_th:.2f}")
    print(f"Heads - Pooled Dispersion Index (Excluding 0s): {pooled_dispersion_nonzero_th:.2f}")


def create_figures(candidates, test_triples, relation_maps, context_triple_store, save_dir):
    """
    Top-level function for creating candidate-related figures.
    Calls individual figure creation functions.

    Args:
        candidates: (M, 3) tensor of candidate triples
        test_triples: (N, 3) tensor of gold test triples
        relation_maps: RelationMaps object
        context_triple_store: List of triples for degree computation
        save_dir: Directory to save the figures
    """
    create_relation_coverage_bar_chart(candidates, test_triples, relation_maps, save_dir)
    create_entity_coverage_bar_chart(candidates, test_triples, context_triple_store, save_dir)
    create_candidates_per_head_by_degree_chart(candidates, context_triple_store, test_triples, save_dir)
    create_test_set_statistics_figure(context_triple_store, test_triples, save_dir)