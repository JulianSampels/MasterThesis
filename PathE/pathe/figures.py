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
    with open(f'{save_dir}/results.csv', 'w') as f:
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
                yticklabels=[f'{b:.1f}' for b in beta_grid], annot=annot_total, fmt='', cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Total Coverage Heatmap')
    plt.savefig(f'{save_dir}/total_coverage_heatmap.svg')  # Saves as SVG image
    plt.close()

    # Plot average recall per group heatmap
    plt.figure(figsize=(8, 6))
    annot_recall = create_annot_matrix(recall_matrix)
    sns.heatmap(recall_matrix, xticklabels=[f'{a:.1f}' for a in alpha_grid], 
                yticklabels=[f'{b:.1f}' for b in beta_grid], annot=annot_recall, fmt='', cmap='viridis', vmin=0, vmax=1)
    plt.xlabel('Alpha')
    plt.ylabel('Beta')
    plt.title('Average Coverage per Group Heatmap')
    plt.savefig(f'{save_dir}/avg_coverage_heatmap.svg')  # Saves as SVG image
    plt.close()

    print(f"Heatmaps and results saved to {save_dir}/")

def create_coverage_vs_size_plot(results, save_dir="./figures", filename="coverage_vs_size.svg"):
    """
    Create a line plot showing total coverage and average recall per group vs. candidate size.
    Saves the plot as an SVG file.

    Args:
        results: List of tuples (candidate_size, total_cov, avg_recall_per_group)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results.sort(key=lambda x: x[0])  # Sort by candidate size
    # Extract data
    candidate_sizes = [r[0] for r in results]
    total_covs = [r[1] for r in results]
    avg_recalls = [r[2] for r in results]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(candidate_sizes, total_covs, label='Total Coverage (Micro)', marker='o', linestyle='-', color='blue')
    plt.plot(candidate_sizes, avg_recalls, label='Avg. Coverage per Group (Macro)', marker='s', linestyle='--', color='red')
    plt.xlabel('Total Candidate Size')
    plt.ylabel('Coverage')
    plt.title('Coverage vs. Candidate Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_relation_coverage_bar_chart(candidates, gold_triples, relation_maps, save_dir="./figures", filename="relation_coverage_bar.svg"):
    """
    Create a bar chart showing coverage (fraction of gold triples covered) for different relation types.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        gold_triples: (N, 3) tensor of gold triples
        relation_maps: RelationMaps object
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique relations from gold triples
    unique_rels = torch.unique(gold_triples[:, 1]).tolist()
    rel_names = [f"Rel_{r}" for r in unique_rels]  # Placeholder names; replace with actual if available
    
    coverage_per_rel = {}
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
    
    # Prepare data for plotting
    rels = list(coverage_per_rel.keys())
    covs = list(coverage_per_rel.values())
    
    # Sort by coverage descending
    sorted_idx = np.argsort(covs)[::-1]
    rels = [rel_names[i] for i in sorted_idx]
    covs = [covs[i] for i in sorted_idx]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(rels)), covs, color='skyblue')
    plt.xlabel('Relation Type')
    plt.ylabel('Coverage')
    plt.title('Coverage per Relation Type')
    plt.xticks(range(len(rels)), rels, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_candidates_per_head_by_degree_chart(candidates, entity_degrees, save_dir="./figures", filename="candidates_per_head_by_degree.svg", degree_bins=10):
    """
    Create a bar chart showing average number of candidates per head, grouped by entity degree bins.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        entity_degrees: Dict[int, int] mapping entity id to its degree (e.g., number of relations)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        degree_bins: Number of bins for degree grouping
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Count candidates per head
    head_counts = defaultdict(int)
    for cand in candidates:
        head = cand[0].item()
        head_counts[head] += 1
    
    # Group by degree
    degrees = [entity_degrees.get(h, 0) for h in head_counts.keys()]
    counts = list(head_counts.values())
    
    # Bin degrees
    if degrees:
        max_deg = max(degrees)
        bins = np.linspace(0, max_deg, degree_bins + 1)
        bin_indices = np.digitize(degrees, bins) - 1
        bin_indices = np.clip(bin_indices, 0, degree_bins - 1)
        
        avg_counts_per_bin = []
        bin_labels = []
        for i in range(degree_bins):
            mask = bin_indices == i
            if mask.any():
                avg_count = np.mean([counts[j] for j in range(len(counts)) if mask[j]])
            else:
                avg_count = 0
            avg_counts_per_bin.append(avg_count)
            bin_labels.append(f"{int(bins[i])}-{int(bins[i+1])}")
    else:
        avg_counts_per_bin = [0] * degree_bins
        bin_labels = [f"0-{int(max_deg/degree_bins)*(i+1)}" for i in range(degree_bins)]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(bin_labels)), avg_counts_per_bin, color='lightgreen')
    plt.xlabel('Entity Degree Bin')
    plt.ylabel('Average Candidates per Head')
    plt.title('Average Candidates per Head by Degree')
    plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def create_entity_coverage_bar_chart(candidates, gold_triples, save_dir="./figures", filename="entity_coverage_bar.svg", top_n=-1):
    """
    Create a bar chart showing coverage (fraction of gold triples covered) for the top N head entities.
    Saves the plot as an SVG file.

    Args:
        candidates: (M, 3) tensor of candidate triples
        gold_triples: (N, 3) tensor of gold triples
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
        top_n: Number of top entities to show
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique heads from gold triples
    unique_heads = torch.unique(gold_triples[:, 0]).tolist()
    
    coverage_per_head = {}
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
    
    # Prepare data for plotting: top N by coverage
    sorted_heads = sorted(coverage_per_head.items(), key=lambda x: x[1], reverse=True)[:top_n]
    heads = [f"Head_{h}" for h, _ in sorted_heads]
    covs = [cov for _, cov in sorted_heads]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(heads)), covs, color='lightcoral')
    plt.xlabel('Head Entity')
    plt.ylabel('Coverage')
    plt.title(f'Coverage per Head Entity (Top {top_n})')
    plt.xticks(range(len(heads)), heads, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")