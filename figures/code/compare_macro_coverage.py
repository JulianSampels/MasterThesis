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

def create_macro_coverage_comparison_plot(results1, results2, results3, save_dir="./figures", filename="macro_coverage_comparison.pgf", relative_divisor=None, loc='center right'):
    """
    Create a line plot comparing Macro Coverage from three different datasets.
    Saves the plot as a PGF file.

    Args:
        results1, results2, results3: Lists of dicts with keys 'candidate_size', 'avg_cov_per_group' (or equivalent)
        save_dir: Directory to save the PGF file
        filename: Name of the output PGF file
        relative_divisor: Optional divisor for relative candidate size on second x-axis
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Sort each results list by candidate_size
    for results in [results1, results2, results3]:
        results.sort(key=lambda x: x['candidate_size'])
    
    # Extract data for Macro Coverage
    candidate_sizes1 = [r['avg_group_count'] for r in results1]
    macro_covs1 = [r['avg_cov_per_group'] for r in results1]
    
    candidate_sizes2 = [r['candidate_size'] for r in results2]
    macro_covs2 = [r['avg_cov_per_group'] for r in results2]  # Adjust key if different
    
    candidate_sizes3 = [r['candidate_size'] for r in results3]
    macro_covs3 = [r['avg_cov_per_group'] for r in results3]  # Adjust key if different
    
    # Create the plot
    WIDTH = 418.25555/72.27
    fig, ax = plt.subplots(figsize=(WIDTH, WIDTH * 0.65))
    
    # Plot Macro Coverage for each dataset
    ax.plot(candidate_sizes1, macro_covs1, label='Coverage Ours', marker='o', markersize=4, linestyle='-', color='seagreen')
    ax.plot(candidate_sizes2, macro_covs2, label='Coverage RETA', marker='s', markersize=4, linestyle='--', color='#4682B4')
    ax.plot(candidate_sizes3, macro_covs3, label='Coverage GFRT', marker='^', markersize=4, linestyle='-.', color='#FF7F50')
    
    ax.set_xlabel('Average Candidate Size')
    ax.set_ylabel('Coverage')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Legend
    ax.legend(loc=loc)
    
    if relative_divisor is not None:
        ax2 = ax.twiny()
        ax2.set_xlabel('Relative Candidate Size')
        ticks = ax.get_xticks()
        ax2.set_xticks(ticks)
        ax2.set_xticklabels([f'{tick / relative_divisor:.1f}' for tick in ticks])
        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 40))
        ax.spines['bottom'].set_position(('outward', 0))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    
    print(f"Plot saved to {save_dir}/{filename}")

def call_plot_from_csvs(csv_path1, csv_path2, csv_path3, plot_func, **kwargs):
    """
    Reads three CSV files and calls the specified plotting function with the data.

    Args:
        csv_path1, csv_path2, csv_path3: Paths to the CSV files.
        plot_func: The plotting function to call.
        **kwargs: Additional keyword arguments to pass to the plotting function.
    """
    df1 = pd.read_csv(csv_path1)
    results1 = df1.to_dict('records')
    
    # Modify results1: replace 'avg_cov_per_group' with candidate_size / 8145 or 6448
    for r in results1:
        r['avg_group_count'] = r['candidate_size'] / 8145  # or 6448 depending on dataset
    
    df2 = pd.read_csv(csv_path2)
    # Assuming column names might be different, e.g., 'macro_coverage' instead of 'avg_cov_per_group'
    # Adjust the key mapping here if needed
    results2 = []
    for row in df2.to_dict('records'):
        results2.append({
            'candidate_size': row['X'],  # Assume this is the same
            'avg_cov_per_group': row.get('Y')  # Adjust key
        })
        print(row.get('Y'))
    
    df3 = pd.read_csv(csv_path3)
    results3 = []
    for row in df3.to_dict('records'):
        results3.append({
            'candidate_size': row['X'],
            'avg_cov_per_group': row.get('Y', row.get('Y', 0))  # Adjust key
        })
    
    plot_func(results1, results2, results3, **kwargs)

# Example usage - replace with actual paths and adjust column names as needed
p1 = "./logs/Final/Fb15k237/pathe2Phases/RelationPTailBce/version_0/figures/coverage_vs_size_results.csv"
p2 = "./RETA_F.csv"  # Replace with actual path
p3 = "./GFRT_F.csv"  # Replace with actual path
call_plot_from_csvs(p1, p2, p3, create_macro_coverage_comparison_plot, filename="FB15k_237_macro_coverage_comparison.pgf", loc='lower right')

# Example usage - replace with actual paths and adjust column names as needed
p1 = "./logs/Final/jf17k/pathe2Phases/RelationBceTailHurdle/version_0/figures/coverage_vs_size_results.csv"
p2 = "./RETA.csv"  # Replace with actual path
p3 = "./GFRT.csv"  # Replace with actual path
call_plot_from_csvs(p1, p2, p3, create_macro_coverage_comparison_plot, filename="JF17k_macro_coverage_comparison.pgf", loc='lower right')