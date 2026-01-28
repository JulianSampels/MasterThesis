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

def create_coverage_vs_total_size_plot(results, save_dir="./figures", filename="coverage_vs_total_size.pgf", relative_divisor=None, loc='center right'):
    """
    Create a line plot showing total coverage, average recall per group, and candidate density vs. candidate size.
    Saves the plot as a PGF file.

    Args:
        results: List of dicts with keys 'candidate_size', 'total_cov', 'avg_cov_per_group', 'pos_density', 'avg_group_density'
        save_dir: Directory to save the PGF file
        filename: Name of the output PGF file
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
    WIDTH = 418.25555/72.27
    fig, ax1 = plt.subplots(figsize=(WIDTH, WIDTH * 0.7))
    
    # Primary axis for coverage metrics
    ax1.plot(candidate_sizes, total_covs, label='Micro Coverage', marker='o', markersize=4, linestyle='-', color='seagreen')
    ax1.plot(candidate_sizes, avg_cov_per_groups, label='Macro Coverage', marker='s', markersize=4, linestyle='--', color='seagreen')
    ax1.set_xlabel('Absolute Candidate Size')
    ax1.set_ylabel('Coverage', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis for density
    ax2 = ax1.twinx()
    ax2.plot(candidate_sizes, pos_densities, label='Micro Density', marker='^', markersize=4, linestyle='-', color='seagreen')
    ax2.plot(candidate_sizes, avg_group_densities, label='Macro Density', marker='v', markersize=4, linestyle='--', color='dodgerblue')
    ax2.set_ylabel('Density', color='dodgerblue')
    ax2.tick_params(axis='y', labelcolor='dodgerblue')
    # ax2.set_ylim(0, 1)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=loc)
    
    # plt.title('Coverage and Density vs. Absolute Candidate Size')
    
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
    plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', pad_inches=0.01)
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
p1 = "./logs/Final/Fb15k237/pathe2Phases/RelationPTailBce/version_0/figures/coverage_vs_size_results.csv"
p2 = "./logs/Final/jf17k/pathe2Phases/RelationBceTailHurdle/version_0/figures/coverage_vs_size_results.csv"
p3 = "./logs/Final/jf17k_filtered/pathe2Phases/RelationBceTailHurdle/version_0/figures/coverage_vs_size_results.csv"
p4 = "./logs/Final/wn18rr/pathe2Phases/RelationPoissonTailBce/version_0/figures/coverage_vs_size_results.csv"
call_plot_from_csv(p1, create_coverage_vs_total_size_plot, filename="fb15k-237_coverage_vs_total_size.pgf", relative_divisor=20438)
call_plot_from_csv(p2, create_coverage_vs_total_size_plot, filename="jf17k_coverage_vs_total_size.pgf", relative_divisor=6823, loc='upper right')
call_plot_from_csv(p3, create_coverage_vs_total_size_plot, filename="jf17k_filtered_coverage_vs_total_size.pgf", relative_divisor=5390)#, loc='upper right')
call_plot_from_csv(p4, create_coverage_vs_total_size_plot, filename="wn18rr_coverage_vs_total_size.pgf", relative_divisor=2924, loc='upper right')
