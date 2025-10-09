import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from collections import defaultdict

def create_heatmaps(results, save_dir="./figures"):
    """
    Create heatmaps for total coverage and average recall per group from grid search results.
    Also saves the results to a CSV file.

    Args:
        results: List of tuples (alpha, beta, temp, total_cov, avg_recall_per_group)
        save_dir: Directory to save the PNG images and results file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save results to CSV
    with open(f'{save_dir}/grid_search_results.csv', 'w') as f:
        f.write("alpha,beta,temp,total_cov,avg_recall_per_group\n")
        for row in results:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
    
    # Extract unique alphas and betas
    alpha_grid = sorted(set([r[0] for r in results]))  # Unique alphas
    beta_grid = sorted(set([r[1] for r in results]))   # Unique betas

    # Create 2D arrays for heatmaps
    total_cov_matrix = np.full((len(beta_grid), len(alpha_grid)), np.nan)
    recall_matrix = np.full((len(beta_grid), len(alpha_grid)), np.nan)

    for alpha, beta, temp, total_cov, avg_recall in results:
        i = beta_grid.index(beta)
        j = alpha_grid.index(alpha)
        total_cov_matrix[i, j] = total_cov
        recall_matrix[i, j] = avg_recall

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
    plt.savefig(f'{save_dir}/total_coverage_heatmap.svg')  # Saves as SVG image
    plt.close()

    # Plot average recall per group heatmap
    plt.figure(figsize=(8, 6))
    annot_recall = create_annot_matrix(recall_matrix)
    sns.heatmap(recall_matrix, xticklabels=[f'{a:.1f}' for a in alpha_grid], 
                yticklabels=[f'{b:.1f}' for b in beta_grid], annot=annot_recall, fmt='', cmap='Greens', vmin=0, vmax=1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Average Coverage per Group Heatmap')
    plt.savefig(f'{save_dir}/avg_coverage_heatmap.svg')  # Saves as SVG image
    plt.close()

    print(f"Heatmaps and results saved to {save_dir}/")

def create_coverage_vs_size_plot(results, save_dir="./figures", filename="coverage_vs_size.svg"):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. candidate size.
    Saves the plot as an SVG file.

    Args:
        results: List of tuples (candidate_size, total_cov, avg_recall_per_group, pos_density)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x[0])  # Sort by candidate size
    # Save results to CSV
    with open(f'{save_dir}/coverage_vs_size_results.csv', 'w') as f:
        f.write("candidate_size,total_cov,avg_recall_per_group,pos_density\n")
        for row in results:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")
    # Extract data
    candidate_sizes = [r[0] for r in results]
    total_covs = [r[1] for r in results]
    avg_recalls = [r[2] for r in results]
    pos_densities = [r[3] for r in results]
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary axis for coverage metrics
    ax1.plot(candidate_sizes, total_covs, label='Total Coverage (Micro)', marker='o', linestyle='-', color='darkgreen')
    ax1.plot(candidate_sizes, avg_recalls, label='Avg. Coverage per Group (Macro)', marker='s', linestyle='--', color='seagreen')
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

def create_relation_coverage_bar_chart(candidates, gold_triples, relation_maps, save_dir="./figures", filename="relation_coverage_bar.svg", num_bins=10):
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
    sorted_indices = np.argsort(covs)
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
    ax1.set_xlabel('Coverage Quantile Bin')
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
    
    plt.title('Average Coverage, Candidates, and Frequency per Relation Coverage Quantile')
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

def create_entity_coverage_bar_chart(candidates, gold_triples, context_triple_store, save_dir="./figures", filename="entity_coverage_bar.svg", num_bins=25):
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
    sorted_indices = np.argsort(covs)
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
    ax1.set_xlabel('Coverage Quantile Bin')
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
    
    plt.title('Average Coverage, Candidates, and Frequency per Head Entity Coverage Quantile')
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

def create_candidate_figures(candidates, test_triples, relation_maps, context_triple_store, save_dir):
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