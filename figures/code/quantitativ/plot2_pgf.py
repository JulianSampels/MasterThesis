import matplotlib
# Set backend to PGF before importing pyplot
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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


def reduce_bins(df, new_bin_count=10):
    """
    Aggregates the dataframe from N bins down to `new_bin_count`.
    Uses array_split to handle cases where len(df) is not divisible by new_bin_count.
    """
    n_rows = len(df)
    
    # --- MODIFIED GROUPING LOGIC ---
    # Instead of integer division, we split the indices into 'new_bin_count' chunks.
    # This handles remainders automatically (e.g. 25 rows -> 5 bins of 3, 5 bins of 2).
    indices = np.arange(n_rows)
    chunks = np.array_split(indices, new_bin_count)
    
    # Create the mapping array
    bin_ids = np.zeros(n_rows, dtype=int)
    for i, chunk in enumerate(chunks):
        bin_ids[chunk] = i
        
    df['new_bin_id'] = bin_ids
    
    # Aggregate metrics by mean
    agg_funcs = {
        'Avg. Coverage per Head (Green)': 'mean',
        'Avg. Candidates per Head (Blue)': 'mean',
        'Avg. Head Frequency (Orange)': 'mean',
        'Entity Degree Quantile Bin': ['first', 'last']
    }
    
    df_agg = df.groupby('new_bin_id').agg(agg_funcs).reset_index()
    
    # Flatten columns
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    
    # Rename columns back
    rename_map = {
        'new_bin_id_': 'new_bin_id',
        'Avg. Coverage per Head (Green)_mean': 'Avg. Coverage per Head (Green)',
        'Avg. Candidates per Head (Blue)_mean': 'Avg. Candidates per Head (Blue)',
        'Avg. Head Frequency (Orange)_mean': 'Avg. Head Frequency (Orange)',
    }
    df_agg.rename(columns=rename_map, inplace=True)
    
    # --- Create simple labels Q1, Q2, ... Q10 ---
    new_labels = [f"Q{i+1}" for i in range(len(df_agg))]
    df_agg['Entity Degree Quantile Bin'] = new_labels
    
    return df_agg

def plot_all_in_one_pgf(csv_file1, csv_file2, save_dir="./figures", filename='combined_all_metrics.pgf'):
    """
    Generates a combined bar/marker chart for Coverage, Candidate Size, and Fact Count
    from two CSV files and saves it as a PGF file.
    """
    # 1. Load Data
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
    except FileNotFoundError:
        print("Error: CSV files not found. Please ensure paths are correct.")
        return
    
    # --- REDUCE BINS (25 -> 10) ---
    df1 = reduce_bins(df1, new_bin_count=10)
    df2 = reduce_bins(df2, new_bin_count=10)

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # 2. Setup Figure and Axes
    # Define width based on the LaTeX textwidth (approx 418pt)
    WIDTH = 418.25555 / 72.27
    
    # Using the dimensions from your previous plot_all_in_one (WIDTH*3 to make it wide enough for many bins)
    fig, ax1 = plt.subplots(figsize=(WIDTH, WIDTH * 0.62))

    # Axis 2: Candidates (Standard Twin Axis)
    ax2 = ax1.twinx()

    # Axis 3: Frequency (Offset Twin Axis)
    ax3 = ax1.twinx()
    # Move the third axis spine to the right
    ax3.spines["right"].set_position(("axes", 1.18))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)

    # 3. Define Plotting Parameters
    bins = df1['Entity Degree Quantile Bin']
    x = np.arange(len(bins))
    width = 0.2  # Width of the bars

    # 4. Plot Data
    
    # --- METRIC 1: COVERAGE (Green, Left Axis) ---
    p1 = ax1.bar(x - 1.5*width, df1['Avg. Coverage per Head (Green)'], width, 
                 label='Coverage (BCE)', color='seagreen', edgecolor='black')
    p2 = ax1.bar(x - 0.5*width, df2['Avg. Coverage per Head (Green)'], width, 
                 label='Coverage (Poisson)', color='#3CB371', edgecolor='black', hatch='//')

    # --- METRIC 2: CANDIDATES (Blue, First Right Axis) ---
    p3 = ax2.bar(x + 0.5*width, df1['Avg. Candidates per Head (Blue)'], width, 
                 label='Mean Candidate Size (BCE)', color='dodgerblue', edgecolor='black')
    p4 = ax2.bar(x + 1.5*width, df2['Avg. Candidates per Head (Blue)'], width, 
                 label='Mean Candidate Size (Poisson)', color='dodgerblue', edgecolor='black', hatch='//')

    # --- METRIC 3: FREQUENCY (Orange, Far Right Axis) ---
    # We only need one legend entry for the support as it's a dataset property
    p6, = ax3.plot(x, df2['Avg. Head Frequency (Orange)'], color='orange', 
                   label='Mean Fact Count', marker='_', markersize=20, linestyle='None', markeredgewidth=2)

    # 5. Configure Axis Labels and Scales
    ax1.set_xlabel('Head Degree Quantile')
    
    # Green Axis (Left)
    ax1.set_ylabel('Macro Coverage', color='seagreen')
    ax1.tick_params(axis='y', labelcolor='seagreen')
    ax1.set_ylim(0, 1.0) 

    # Blue Axis (Right 1)
    ax2.set_ylabel('Mean Candidate Set Size', color='dodgerblue')
    ax2.tick_params(axis='y', labelcolor='dodgerblue')
    ax2.set_ylim(0, 400) 

    # Orange Axis (Right 2)
    ax3.set_ylabel('Mean Fact Count', color='#D35400')
    ax3.tick_params(axis='y', labelcolor='#D35400')
    ax3.set_ylim(0, 8) 

    # X-Axis Formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
    
    # Set x-limits to minimize empty space
    ax1.set_xlim(-0.6, len(bins) - 0.4)

    # 6. Combined Legend
    handles = [p1, p2, p3, p4, p6]
    labels = [h.get_label() for h in handles]
    
    # Place legend at top left
    ax1.legend(handles, labels, loc='upper left', ncol=1, frameon=True)

    # 7. Save
    plt.tight_layout()
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.01)
    print(f"Graph saved as {output_path}")

# --- Execution ---
p1 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/bcebceFB15k-237.csv"
p2 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/PPFB15k-237.csv"

# Make sure to update paths or file names as needed
plot_all_in_one_pgf(p1, p2, filename="quantile_bins_bce_vs_poisson.pgf")