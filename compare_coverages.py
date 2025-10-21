import pandas as pd
import matplotlib.pyplot as plt
import os

def create_compare_coverage_vs_avg_group_size_plot(csv_paths, labels=None, save_dir="./figures", filename="compare_coverage_vs_avg_group_size.svg"):
    """
    Create a line plot comparing total coverage, average recall per group, and positive density vs. Candidate Size
    for multiple datasets.
    Saves the plot as an SVG file.

    Args:
        csv_paths: List of paths to the CSV files
        labels: List of labels for the datasets (optional, defaults to Dataset 1, Dataset 2, etc.)
        save_dir: Directory to save the SVG file
        filename: Name of the output SVG file
    """
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(csv_paths))]
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.sort_values('candidate_size')
        dfs.append(df)
    
    # Create the plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Colors for different datasets
    colors_cov = ['darkgreen', 'darkblue', 'darkred', 'darkorange', 'darkmagenta', "darkcyan"]
    colors_density = ['red', 'blue', 'green', 'orange', 'magenta', 'cyan']

    # Primary axis for coverage metrics
    for i, (df, label) in enumerate(zip(dfs, labels)):
        avg_group_counts = df['candidate_size']
        avg_cov_per_groups = df['avg_cov_per_group']
        ax1.plot(avg_group_counts, avg_cov_per_groups, label=f'Total Coverage (Micro) - {label}', marker='o', linestyle='-', color=colors_cov[i % len(colors_cov)])
    
    ax1.set_xlabel('Total Candidate Size')
    ax1.set_ylabel('Coverage')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 1)
    # ax1.set_xlim(0, 3e6)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    for i, (df, label) in enumerate(zip(dfs, labels)):
        avg_group_counts = df['candidate_size']
        pos_densities = df['pos_density']
        ax2.plot(avg_group_counts, pos_densities, label=f'Positives Density - {label}', marker='x', linestyle=':', color=colors_density[i % len(colors_density)])
    
    ax2.set_ylabel('Density', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Comparison of Coverage and Density vs. Candidate Size')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}')
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

if __name__ == "__main__":
    csv_paths = [
        "figures/pathe2Phases/FinalRunWithAccGrad1NoRpTpWeights/global_with_tail/global_joint/coverage_vs_avg_group_size_results.csv",
        "/home/juliansampels/Masterarbeit/MasterThesis/figures/pathe2Phases/testsCompareCandidates/global_with_tail/global_joint/coverage_vs_size_results.csv",
        "figures/pathe2Phases/FinalRunWithAccGrad8/global_with_tail/global_joint/coverage_vs_avg_group_size_results.csv",
        "figures/pathe2Phases/FinalRunWithAccGrad1/global_with_tail/global_joint/coverage_vs_avg_group_size_results.csv",
        # "figures/candidate_grid_search/testsCompareCandidates/global_with_tail/global_joint/coverage_vs_size_results.csv",
        "figures/pathe2Phases/FinalRunWithCountingWeights/global_with_tail/global_joint/coverage_vs_size_results.csv",
        "figures/pathe2Phases/FinalRunWithAccGrad1EqualCountingWeigts/global_with_tail/global_joint/coverage_vs_size_results.csv",
    ]
    labels = ["FinalRunWithAccGrad1NoRpTpWeights", "Compare Candidates without tails?", "Final Run with Acc Grad 8", "Final Run with Acc Grad 1", #"candidate_grid_search", 
              "Final Run with Counting Weights", "Final Run with Acc Grad 1 Equal Counting Weights"]
    create_compare_coverage_vs_avg_group_size_plot(csv_paths, labels)