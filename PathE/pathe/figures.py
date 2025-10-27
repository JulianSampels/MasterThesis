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

def create_coverage_vs_size_plot(results, save_dir="./figures"):
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['total_cov'])  # Sort by total coverage
    # Save results to CSV using pandas for simplicity
    df = pd.DataFrame(results)
    df.to_csv(f'{save_dir}/coverage_vs_size_results.csv', index=False)
    create_coverage_vs_total_size_plot(results, save_dir=save_dir)
    create_coverage_vs_avg_group_size_plot(results, save_dir=save_dir)


def create_coverage_vs_total_size_plot(results, save_dir="./figures", filename="coverage_vs_total_size.svg"):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. candidate size.
    Saves the plot as an SVG file.

    Args:
        results: List of dicts with keys 'candidate_size', 'total_cov', 'avg_cov_per_group', 'pos_density'
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['candidate_size'])  # Sort by candidate size
    # Extract data
    candidate_sizes = [r['candidate_size'] for r in results]  # Total candidate size
    total_covs = [r['total_cov'] for r in results]
    avg_cov_per_groups = [r['avg_cov_per_group'] for r in results]
    pos_densities = [r['pos_density'] for r in results]
    # avg_group_counts = [r['avg_group_count'] for r in results]
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis for coverage metrics
    ax1.plot(candidate_sizes, total_covs, label='Total Coverage (Micro)', marker='o', linestyle='-', color='darkgreen')
    ax1.plot(candidate_sizes, avg_cov_per_groups, label='Avg. Coverage per Group (Macro)', marker='s', linestyle='--', color='seagreen')
    ax1.set_xlabel('Total Candidate Size')
    ax1.set_ylabel('Coverage')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(candidate_sizes, pos_densities, label='Total Positives Density', marker='^', linestyle=':', color='blue')
    ax2.set_ylabel('Density', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Coverage and Density vs. Candidate Size')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot and results saved to {save_dir}/")

def create_coverage_vs_avg_group_size_plot(results, save_dir="./figures", filename="coverage_vs_avg_group_size.svg"):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. avg candidate size per group.
    Saves the plot as an SVG file.

    Args:
        results: List of dicts with keys 'candidate_size', 'total_cov', 'avg_cov_per_group', 'pos_density', 'avg_group_count'
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x['avg_group_count'])  # Sort by avg candidate count per group
    # Extract data
    avg_group_counts = [r['avg_group_count'] for r in results]  # Avg candidate size per group
    total_covs = [r['total_cov'] for r in results]
    avg_cov_per_groups = [r['avg_cov_per_group'] for r in results]
    pos_densities = [r['pos_density'] for r in results]
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis for coverage metrics
    ax1.plot(avg_group_counts, total_covs, label='Total Coverage (Micro)', marker='o', linestyle='-', color='darkgreen')
    ax1.plot(avg_group_counts, avg_cov_per_groups, label='Avg. Coverage per Group (Macro)', marker='s', linestyle='--', color='seagreen')
    ax1.set_xlabel('Avg Candidate Size per Group')
    ax1.set_ylabel('Coverage')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(avg_group_counts, pos_densities, label='Total Positives Density', marker='^', linestyle=':', color='blue')
    ax2.set_ylabel('Density', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Coverage and Density vs. Avg Candidate Size per Group')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot and results saved to {save_dir}/")

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

def create_tail_occurrence_per_head_figure(triples, save_dir="./figures", filename="tail_occurrence_per_head.svg", max_samples=1000000):
    """
    Create a histogram showing the distribution of the total number of tail occurrences per head entity (out-degree).
    This provides statistics on the occurrence count of tails for heads in the knowledge graph.

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
    
    # Compute total tail occurrences per head (out-degree) - vectorized
    head_to_tail_count = defaultdict(int)
    for triple in triples:
        head_to_tail_count[triple[0].item()] += 1
    
    # Get all unique tails - vectorized
    all_tails = torch.unique(triples[:, 2]).tolist()
    
    # Compute per-head per-tail counts - optimize: only compute averages, not full dict
    head_to_avg_tail = {}
    for head in head_to_tail_count.keys():
        head_triples = triples[triples[:, 0] == head]
        if head_triples.size(0) > 0:
            tail_counts = torch.bincount(head_triples[:, 2], minlength=len(all_tails))
            avg = tail_counts.float().mean().item()
            head_to_avg_tail[head] = avg
    
    # Compute averages
    avg_tail_per_head_list = list(head_to_avg_tail.values())
    avg_tail = sum(avg_tail_per_head_list) / len(avg_tail_per_head_list) if avg_tail_per_head_list else 0
    
    # For excluding zero: average of averages where counts > 0
    nonzero_avg_tail_per_head_list = []
    for head in head_to_avg_tail.keys():
        head_triples = triples[triples[:, 0] == head]
        tail_counts = torch.bincount(head_triples[:, 2], minlength=len(all_tails))
        nonzero_counts = tail_counts[tail_counts > 0].float()
        if nonzero_counts.numel() > 0:
            nonzero_avg_tail_per_head_list.append(nonzero_counts.mean().item())
    
    avg_tail_nonzero = sum(nonzero_avg_tail_per_head_list) / len(nonzero_avg_tail_per_head_list) if nonzero_avg_tail_per_head_list else 0
    
    print(f"Average tail occurrences per head: {avg_tail:.2f}")
    print(f"Average tail occurrences per head (excluding 0): {avg_tail_nonzero:.2f}")
    
    # Get tail_counts for histogram - sample if too large
    tail_counts = []
    for head in head_to_tail_count.keys():
        head_triples = triples[triples[:, 0] == head]
        tail_counts.extend(torch.bincount(head_triples[:, 2], minlength=len(all_tails)).tolist())
    
    if len(tail_counts) > max_samples:
        import random
        tail_counts = random.sample(tail_counts, max_samples)
        print(f"Sampled {max_samples} tail counts for histogram to avoid OOM.")
    
    print(f"Total number of tail occurrence counts collected: {len(tail_counts)}, max: {max(tail_counts)}, min: {min(tail_counts)}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    min_count = min(tail_counts)
    max_count = max(tail_counts)
    bins = np.arange(min_count - 0.5, max_count + 1.5, 1)
    plt.hist(tail_counts, bins=bins, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Number of Tail Occurrences per Head')
    plt.ylabel('Number of Heads')
    plt.title('Distribution of Tail Occurrence Counts per Head')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xticks(np.arange(min_count, max_count + 1, 1))
    
    # Add text with averages
    plt.text(0.7, 0.9, f'Avg tails/head: {avg_tail:.2f}\nAvg tails/head (non-zero): {avg_tail_nonzero:.2f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Figure saved to {save_dir}/{filename}")

def create_relation_occurrence_figure(triples, save_dir="./figures", filename="relation_occurrence.svg", max_samples=1000000):
    """
    Create a histogram showing the distribution of the occurrence count of each relation.
    This provides statistics on how frequently each relation appears in the knowledge graph.

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
    
    # Compute occurrence count per relation - vectorized
    relation_counts = defaultdict(int)
    for triple in triples:
        relation_counts[triple[1].item()] += 1
    
    # Get all unique relations - vectorized
    all_relations = torch.unique(triples[:, 1]).tolist()
    
    # Compute per-head relation counts - optimize: only compute averages
    head_to_avg_rel = {}
    for head in torch.unique(triples[:, 0]).tolist():
        head_triples = triples[triples[:, 0] == head]
        if head_triples.size(0) > 0:
            rel_counts = torch.bincount(head_triples[:, 1], minlength=len(all_relations))
            avg = rel_counts.float().mean().item()
            head_to_avg_rel[head] = avg
    
    # Compute averages
    avg_rel_per_head_list = list(head_to_avg_rel.values())
    avg_rel = sum(avg_rel_per_head_list) / len(avg_rel_per_head_list) if avg_rel_per_head_list else 0
    
    # For excluding zero: average of averages where counts > 0
    nonzero_avg_rel_per_head_list = []
    for head in head_to_avg_rel.keys():
        head_triples = triples[triples[:, 0] == head]
        rel_counts = torch.bincount(head_triples[:, 1], minlength=len(all_relations))
        nonzero_counts = rel_counts[rel_counts > 0].float()
        if nonzero_counts.numel() > 0:
            nonzero_avg_rel_per_head_list.append(nonzero_counts.mean().item())
    
    avg_rel_nonzero = sum(nonzero_avg_rel_per_head_list) / len(nonzero_avg_rel_per_head_list) if nonzero_avg_rel_per_head_list else 0
    
    print(f"Average relation occurrences per head: {avg_rel:.2f}")
    print(f"Average relation occurrences per head (excluding 0): {avg_rel_nonzero:.2f}")
    
    # Get counts for histogram - sample if too large
    counts = []
    for head in head_to_avg_rel.keys():
        head_triples = triples[triples[:, 0] == head]
        counts.extend(torch.bincount(head_triples[:, 1], minlength=len(all_relations)).tolist())
    # counts = [c for c in counts if c > 0]  # Only consider non-zero counts
    if len(counts) > max_samples:
        import random
        counts = random.sample(counts, max_samples)
        print(f"Sampled {max_samples} relation counts for histogram to avoid OOM.")
    
    print(f"Total number of relation occurrence counts collected: {len(counts)}, max: {max(counts)}, min: {min(counts)}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    min_count = min(counts)
    max_count = max(counts)
    bins = np.arange(min_count - 0.5, max_count + 1.5, 1)
    plt.hist(counts, bins=bins, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Number of Occurrences per Relation per Head')
    plt.ylabel('Number of Relations')
    plt.title('Distribution of Relation Occurrence Counts')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    # plt.xticks(np.arange(min_count, max_count + 1, 1))
    
    # Add text with averages for relations per head
    plt.text(0.7, 0.9, f'Avg rels/head: {avg_rel:.2f}\nAvg rels/head (non-zero): {avg_rel_nonzero:.2f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Figure saved to {save_dir}/{filename}")


def plot_relation_count_dispersion(triples, save_dir="./figures", filename="relation_dispersion.svg"):
    """
    Analyzes if relation counts per relation type follow a Poisson distribution by plotting variance vs. mean.
    For each relation type, computes the mean and variance of its occurrence counts across all heads (including 0 for heads without that relation).

    Args:
        triples (torch.Tensor): (N, 3) tensor of (head, relation, tail).
        save_dir (str): Directory to save the figure.
        filename (str): Name of the output SVG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    rel_to_head_counts = defaultdict(lambda: defaultdict(int))
    all_relations = set()
    all_heads = set()

    for h, r, t in triples:
        h, r = h.item(), r.item()
        rel_to_head_counts[r][h] += 1
        all_relations.add(r)
        all_heads.add(h)

    means, variances = [], []
    means_nonzero, variances_nonzero = [], []
    num_heads = len(all_heads)

    for rel in all_relations:
        # Get counts for this relation, including 0 for heads not present
        counts = [rel_to_head_counts[rel].get(h, 0) for h in all_heads]
        
        if len(counts) > 1:
            means.append(np.mean(counts))
            variances.append(np.var(counts))
            
            # Non-zero counts
            counts_nonzero = [c for c in counts if c > 0]
            if len(counts_nonzero) > 1:
                means_nonzero.append(np.mean(counts_nonzero))
                variances_nonzero.append(np.var(counts_nonzero))

    if not means:
        print("No data to plot for dispersion.")
        return

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(means, variances, alpha=0.5, color='red', label='Per-Relation Type Mean vs. Variance (including 0s)')
    plt.scatter(means_nonzero, variances_nonzero, alpha=0.5, color='blue', label='Per-Relation Type Mean vs. Variance (excluding 0s)')
    
    # Add y=x line for reference
    max_val = max(max(means + means_nonzero), max(variances + variances_nonzero))
    plt.plot([0, max_val], [0, max_val], 'k--', label='y=x (Poisson ideal)')
    
    plt.xlabel('Mean of Head Counts per Relation Type')
    plt.ylabel('Variance of Head Counts per Relation Type')
    plt.title('Relation Count Dispersion per Relation Type')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # plt.xscale('log')
    # plt.yscale('log')
    
    # Calculate and display overall dispersion indices
    overall_dispersion = np.mean(variances) / np.mean(means) if np.mean(means) > 0 else float('nan')
    overall_dispersion_nonzero = np.mean(variances_nonzero) / np.mean(means_nonzero) if np.mean(means_nonzero) > 0 else float('nan')
    textstr = f'Including 0s: {overall_dispersion:.2f}\nExcluding 0s: {overall_dispersion_nonzero:.2f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Dispersion plot saved to {os.path.join(save_dir, filename)}")
    print(f"Including 0s dispersion index: {overall_dispersion:.2f}")
    print(f"Excluding 0s dispersion index: {overall_dispersion_nonzero:.2f}")


def plot_tail_count_dispersion(triples, save_dir="./figures", filename="tail_dispersion.svg"):
    """
    Analyzes if tail counts per tail follow a Poisson distribution by plotting variance vs. mean.
    For each tail, computes the mean and variance of its occurrence counts across all heads (including 0 for heads without that tail).

    Args:
        triples (torch.Tensor): (N, 3) tensor of (head, relation, tail).
        save_dir (str): Directory to save the figure.
        filename (str): Name of the output SVG file.
    """
    os.makedirs(save_dir, exist_ok=True)

    tail_to_head_counts = defaultdict(lambda: defaultdict(int))
    all_tails = set()
    all_heads = set()

    for h, r, t in triples:
        h, t = h.item(), t.item()
        tail_to_head_counts[t][h] += 1
        all_tails.add(t)
        all_heads.add(h)

    means, variances = [], []
    means_nonzero, variances_nonzero = [], []
    num_heads = len(all_heads)

    for tail in all_tails:
        # Get counts for this tail, including 0 for heads not present
        counts = [tail_to_head_counts[tail].get(h, 0) for h in all_heads]
        
        if len(counts) > 1:
            means.append(np.mean(counts))
            variances.append(np.var(counts))
            
            # Non-zero counts
            counts_nonzero = [c for c in counts if c > 0]
            if len(counts_nonzero) > 1:
                means_nonzero.append(np.mean(counts_nonzero))
                variances_nonzero.append(np.var(counts_nonzero))

    if not means:
        print("No data to plot for dispersion.")
        return

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(means, variances, alpha=0.5, color='red', label='Per-Tail Mean vs. Variance (including 0s)')
    plt.scatter(means_nonzero, variances_nonzero, alpha=0.5, color='blue', label='Per-Tail Mean vs. Variance (excluding 0s)')
    
    # Add y=x line for reference
    max_val = max(max(means + means_nonzero), max(variances + variances_nonzero))
    plt.plot([0, max_val], [0, max_val], 'k--', label='y=x (Poisson ideal)')
    
    plt.xlabel('Mean of Head Counts per Tail')
    plt.ylabel('Variance of Head Counts per Tail')
    plt.title('Tail Count Dispersion per Tail')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.axis('equal')
    
    # Calculate and display overall dispersion indices
    overall_dispersion = np.mean(variances) / np.mean(means) if np.mean(means) > 0 else float('nan')
    overall_dispersion_nonzero = np.mean(variances_nonzero) / np.mean(means_nonzero) if np.mean(means_nonzero) > 0 else float('nan')
    textstr = f'Including 0s: {overall_dispersion:.2f}\nExcluding 0s: {overall_dispersion_nonzero:.2f}'
    plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Dispersion plot saved to {os.path.join(save_dir, filename)}")
    print(f"Including 0s dispersion index: {overall_dispersion:.2f}")
    print(f"Excluding 0s dispersion index: {overall_dispersion_nonzero:.2f}")


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