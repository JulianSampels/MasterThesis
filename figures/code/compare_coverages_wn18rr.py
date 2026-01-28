from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import os

# --- 1. Define Custom Handler for Centered Text ---
class TextHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Draw the text passed as 'orig_handle' in the center of the legend box
        t = mtext.Text(xdescent + width/2, ydescent + height/2, orig_handle, 
                       ha='center', va='center', fontsize=fontsize)
        return [t]

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'sans-serif',
    'text.usetex': False,  # Changed to False (PGF backend handles LaTeX generation internally)
    'pgf.rcfonts': False,
    'font.size': 11,
    'pgf.preamble': "\n".join([
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        # --- FIX 1: Define \mathdefault to prevent the crash ---
        r"\def\mathdefault#1{#1}",
        # --- FIX 2: Force Sans-Serif Math (Numbers) ---
        r"\usepackage{sansmath}",
        r"\sansmath",
        r"\usepackage{amsmath}"
    ])
})

def create_compare_coverage_vs_avg_group_size_plot(csv_paths, labels=None, save_dir="./figures", filename="compare_coverage_vs_avg_group_size.pgf", relative_divisor=None):
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(csv_paths))]
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the data
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.sort_values('candidate_size')
        dfs.append(df)
    
    WIDTH = 418.25555/72.27
    # Reset height to normal ratio since legend is inside
    fig, ax1 = plt.subplots(figsize=(WIDTH, WIDTH * 1.15)) 
    
    colors = [
        "#2FC700", "#E69F00", "#56B4E9", "#009E73", 
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7", 
        "#999999", "#332288", "#882255", "#117733"
    ]

    # Plot Coverage
    lines_cov = []
    for i, (df, label_parts) in enumerate(zip(dfs, labels)):
        l, = ax1.plot(df['candidate_size'], df['avg_cov_per_group'], 
                      marker='o', markersize=4, linestyle='-', color=colors[i % len(colors)])
        lines_cov.append(l)
    
    ax1.set_xlabel('Absolute Candidate Size')
    ax1.set_ylabel('Macro Coverage')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot Density
    ax2 = ax1.twinx()
    lines_den = []
    for i, (df, label_parts) in enumerate(zip(dfs, labels)):
        l, = ax2.plot(df['candidate_size'], df['avg_group_density'], 
                      marker='x', markersize=4, linestyle=':', color=colors[i % len(colors)])
        lines_den.append(l)
    
    ax2.set_ylabel('Macro Density', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # --- NEW LEGEND LOGIC (Inside Plot) ---
    
    # 1. Initialize Columns with Headers (passed as strings for TextHandler)
    h_rel = [r"\textbf{Relation}"]
    h_ent = [r"\textbf{Entity}"]
    h_cov = [r"\textbf{Coverage}"]
    h_den = [r"\textbf{Density}"]
    
    # 2. Fill Data Rows
    for label_parts, l_cov, l_den in zip(labels, lines_cov, lines_den):
        if isinstance(label_parts, tuple):
            r_txt, e_txt = label_parts
        else:
            r_txt, e_txt = str(label_parts), ""
            
        h_rel.append(r_txt)
        h_ent.append(e_txt)
        h_cov.append(l_cov)
        h_den.append(l_den)

    # 3. Combine Columns
    final_handles = h_rel + h_ent + h_cov + h_den
    final_labels  = [""] * len(final_handles) # Labels empty because text is in handles
    
    # 4. Create Legend
    ax1.legend(
        final_handles,
        final_labels,
        ncol=4,
        loc='upper center', # <-- Placed inside (try 'best' or 'center right' if this covers data)
        # bbox_to_anchor=(0.99, 0.51),
        handler_map={str: TextHandler()}, 
        # handlelength=4.5,  # Width of the columns (adjust if text is clipped)
        columnspacing=2.5, 
        labelspacing=0.4,
        # borderpad= 1,
        handletextpad=0.0,
        frameon=False
        # prop={'size': 8},  # Smaller font to fit inside plot
        # framealpha=0.9     # Semi-transparent background so lines show through faintly
    )
    
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
    
    # 2. Add MANUAL Background Box
    rect = Rectangle(
        (0.409e6, 0.828),      # (x, y) Bottom-Left corner
        1.93e6,                # width
        0.16,               # height
        # transform=ax1.transAxes, 
        facecolor='white',
        edgecolor='lightgray',     
        linewidth=0.75,         
        alpha=0.8,
        zorder=5,
    )
    ax1.add_patch(rect)


    plt.tight_layout()
    plt.savefig(f'{save_dir}/{filename}', bbox_inches='tight', pad_inches=0.01)
    plt.close()
    print(f"Plot saved to {save_dir}/{filename}")

if __name__ == "__main__":
    # ... (Your main block remains identical) ...
    csv_paths = [
        "./logs/Final/wn18rr/pathe2Phases/RelationBceTailBce/version_0/figures/coverage_vs_size_results.csv",
        "./logs/Final/wn18rr/pathe2Phases/RelationHTailBce/version_0/figures/coverage_vs_size_results.csv",
        "./logs/Final/wn18rr/pathe2Phases/RelationPoissonTailBce/version_0/figures/coverage_vs_size_results.csv"
    ]
    labels = [
        ("BCE", "BCE"),
        ("Hurdle", "BCE"),
        ("Poisson", "BCE")
    ]
    create_compare_coverage_vs_avg_group_size_plot(csv_paths, labels, filename="wn18rr_compare_macro.pgf", relative_divisor=2924)