import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

def plot_all_in_one(csv_file1, csv_file2, output_filename='combined_all_metrics.png'):
    # 1. Load Data
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
    except FileNotFoundError:
        print("Error: CSV files not found. Please ensure data_1.csv and data_2.csv exist.")
        return

    # 2. Setup Figure and Axes
    # We need a host figure and two parasitic axes for the different scales
    WIDTH = 418.25555/72.27
    fig, ax1 = plt.subplots(figsize=(WIDTH*3, 3*WIDTH * 0.7))
    # plt.subplots_adjust(right=0.85) # Reserve space for the third axis on the right

    # Axis 2: Candidates (Standard Twin Axis)
    ax2 = ax1.twinx()

    # Axis 3: Frequency (Offset Twin Axis)
    ax3 = ax1.twinx()
    # Move the third axis spine to the right by 20% of the plot width
    ax3.spines["right"].set_position(("axes", 1.07))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)

    # 3. Define Plotting Parameters
    bins = df1['Entity Degree Quantile Bin']
    x = np.arange(len(bins))
    width = 0.2 # Narrower width to fit 4 bars per tick

    # 4. Plot Data
    
    # --- METRIC 1: COVERAGE (Green, Left Axis) ---
    # Bars are shifted to the left of the center tick
    # Data 1 (Dark Green)
    p1 = ax1.bar(x - 1.5*width, df1['Avg. Coverage per Head (Green)'], width, 
                 label='Coverage (BCE, BCE)', color='seagreen', edgecolor='black')
    # Data 2 (Light Green)
    p2 = ax1.bar(x - 0.5*width, df2['Avg. Coverage per Head (Green)'], width, 
                 label='Coverage (Poisson, Poisson)', color='#3CB371', edgecolor='black', hatch='//')

    # --- METRIC 2: CANDIDATES (Blue, First Right Axis) ---
    # Bars are shifted to the right of the center tick
    # Data 1 (Dark Blue)
    p3 = ax2.bar(x + 0.5*width, df1['Avg. Candidates per Head (Blue)'], width, 
                 label='Candidate Size (BCE, BCE)', color='dodgerblue', edgecolor='black')
    # Data 2 (Light Blue)
    p4 = ax2.bar(x + 1.5*width, df2['Avg. Candidates per Head (Blue)'], width, 
                 label='Candidate Size (Poisson, Poisson)', color='dodgerblue', edgecolor='black', hatch='//')

    # --- METRIC 3: FREQUENCY (Orange, Far Right Axis) ---
    # Lines are centered on the tick
    # Data 1 (Solid Orange Line)
    # p5, = ax3.plot(x, df1['Avg. Head Frequency (Orange)'], color='#D21518', 
    #                label='File 1: Frequency', marker='o', linewidth=2.5)
    # Data 2 (Dashed Orange Line)
    p6, = ax3.plot(x, df2['Avg. Head Frequency (Orange)'], color='orange', 
                   label='Mean Fact Count', marker='_', markersize=30, linestyle='None', markeredgewidth=2)

    # 5. Configure Axis Labels and Scales
    ax1.set_xlabel('Entity Degree Quantiles', fontsize=14)
    
    # Green Axis (Left)
    ax1.set_ylabel('Macro Coverage', color='seagreen', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='seagreen', labelsize=10)
    ax1.set_ylim(0, 1.0) # Fixed scale 0-1

    # Blue Axis (Right 1)
    ax2.set_ylabel('Mean Candidate Set Size', color='dodgerblue', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='dodgerblue', labelsize=10)
    # Auto-scale usually works, but we can set a max to align visually if needed.
    # Data 2 max is ~530, so 600 is a safe upper limit.
    ax2.set_ylim(0, 550) 

    # Orange Axis (Right 2)
    ax3.set_ylabel('Mean Fact Count', color='#D35400', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='#D35400', labelsize=10)
    ax3.set_ylim(0, 10) # Max frequency is ~9.5

    # X-Axis Formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, rotation=45, ha='right', fontsize=10)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.5)

    # 6. Combined Legend
    # plt.title('Coverage, Candidates, and Frequency', fontsize=16, pad=20)
    
    # Collect all handles for a unified legend
    handles = [p1, p2, p3, p4, p6]
    labels = [h.get_label() for h in handles]
    
    # Place legend at top left
    ax1.legend(handles, labels, loc='upper left', ncol=3, frameon=True, fontsize=10)

    # 7. Save
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Graph saved as {output_filename}")

# --- Execution ---
# Ensure you have 'data_1.csv' and 'data_2.csv' in the same directory.
# plot_all_in_one('data_1.csv', 'data_2.csv')
p1 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/bcebceFB15k-237.csv"
p2 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/PPFB15k-237.csv"
plot_all_in_one(p1, p2)