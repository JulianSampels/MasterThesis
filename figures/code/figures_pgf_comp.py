import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from collections import defaultdict
import pandas as pd

# --- Configure Matplotlib for PGF/Sans-Serif ---
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'sans-serif',
    'text.usetex': False,  # PGF backend handles LaTeX generation internally
    'pgf.rcfonts': False,
    'font.size': 11,
    'pgf.preamble': "\n".join([
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\def\mathdefault#1{#1}",
        r"\usepackage{sansmath}",
        r"\sansmath",
        r"\usepackage{amsmath}"
    ])
})

def create_coverage_vs_total_size_plot(results1, results2, save_dir="./figures", filename="coverage_vs_total_size.pgf", labels=('Dataset 1', 'Dataset 2')):
    """
    Create a line plot comparing Macro Coverage and Macro Density from two datasets.
    Saves the plot as a PGF file.

    Args:
        results1, results2: Lists of dicts with keys 'candidate_size', 'total_cov', 'pos_density'
        save_dir: Directory to save the PGF file
        filename: Name of the output PGF file
        labels: Tuple of labels for the two datasets
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    results1.sort(key=lambda x: x['candidate_size'])  # Sort by candidate size
    results2.sort(key=lambda x: x['candidate_size'])
    # Extract data and compute relative sizes
    relative_sizes1 = [r['candidate_size'] / 6823 for r in results1]  # JF17K non-filtered
    total_covs1 = [r['avg_cov_per_group'] for r in results1]
    pos_densities1 = [r['avg_group_density'] for r in results1]
    
    relative_sizes2 = [r['candidate_size'] / 5390 for r in results2]  # JF17K filtered
    total_covs2 = [r['avg_cov_per_group'] for r in results2]
    pos_densities2 = [r['avg_group_density'] for r in results2]
    
    # Create the plot with dual y-axes
    WIDTH = 418.25555/72.27
    fig, ax1 = plt.subplots(figsize=(WIDTH, WIDTH * 0.7))
    
    # Primary axis for coverage metrics
    ax1.plot(relative_sizes2, total_covs2, label=f'Macro Coverage ({labels[1]})', marker='o', markersize=4, linestyle='-', color='seagreen')
    ax1.plot(relative_sizes1, total_covs1, label=f'Macro Coverage ({labels[0]})', marker='o', markersize=4, linestyle=':', color='seagreen')
    ax1.set_xlabel('Relative Candidate Size')
    ax1.set_ylabel('Macro Coverage', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(relative_sizes2, pos_densities2, label=f'Macro Density ({labels[1]})', marker='^', markersize=4, linestyle='-', color='dodgerblue')
    ax2.plot(relative_sizes1, pos_densities1, label=f'Macro Density ({labels[0]})', marker='^', markersize=4, linestyle=':', color='dodgerblue')
    ax2.set_ylabel('Macro Density', color='dodgerblue')
    ax2.tick_params(axis='y', labelcolor='dodgerblue')
    # ax2.set_ylim(0, 1)
    
    # Separate legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='center right', bbox_to_anchor=(1.0, 0.441))
    
    # plt.title('Coverage and Density vs. Absolute Candidate Size')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    
    print(f"Plot and results saved to {save_dir}/{filename}")

def call_plot_from_csv(csv_path1, csv_path2, plot_func, labels=('Dataset 1', 'Dataset 2'), **kwargs):
    """
    Reads two CSV files and calls the specified plotting function with the data.

    Args:
        csv_path1, csv_path2: Paths to the CSV files containing results.
        plot_func: The plotting function to call (e.g., create_coverage_vs_total_size_plot).
        labels: Tuple of labels for the datasets.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    df1 = pd.read_csv(csv_path1)
    results1 = df1.to_dict('records')
    df2 = pd.read_csv(csv_path2)
    results2 = df2.to_dict('records')
    plot_func(results1, results2, labels=labels, **kwargs)
p2 = "./logs/Final/jf17k/pathe2Phases/RelationBceTailHurdle/version_0/figures/coverage_vs_size_results.csv"
p3 = "./logs/Final/jf17k_filtered/pathe2Phases/RelationBceTailHurdle/version_0/figures/coverage_vs_size_results.csv"
call_plot_from_csv(p2, p3, create_coverage_vs_total_size_plot, labels=('JF17K', 'JF17K-filtered'), filename="jf17k_vs_jf17k_filtered_macro.pgf")
