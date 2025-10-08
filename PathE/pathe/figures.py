import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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