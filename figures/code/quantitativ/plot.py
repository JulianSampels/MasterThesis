import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_combined_metrics(csv_file1, csv_file2, output_filename1='figure1_coverage.png', output_filename2='figure2_candidates.png'):
    """
    Reads two CSV files with the structure:
    'Entity Degree Quantile Bin', 'Avg. Coverage per Head (Green)',
    'Avg. Candidates per Head (Blue)', 'Avg. Head Frequency (Orange)'

    Generates two plots:
    1. Coverage per Head (grouped bars) + Head Frequency (lines).
    2. Candidates per Head (grouped bars) + Head Frequency (lines).
    """

    # 1. Load Data
    try:
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Basic validation
    required_cols = [
        'Entity Degree Quantile Bin',
        'Avg. Coverage per Head (Green)',
        'Avg. Candidates per Head (Blue)',
        'Avg. Head Frequency (Orange)'
    ]
    
    for df, name in zip([df1, df2], ['File 1', 'File 2']):
        # Clean column names just in case of trailing spaces
        df.columns = df.columns.str.strip()
        if not all(col in df.columns for col in required_cols):
            print(f"Error: {name} does not have the correct column structure.")
            print(f"Expected: {required_cols}")
            return

    # Prepare X-axis
    bins = df1['Entity Degree Quantile Bin']
    x = np.arange(len(bins))
    width = 0.35  # Width of the bars

    # ==========================================
    # FIGURE 1: Coverage per Head (Grouped) + Head Frequency
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(18, 8))

    # -- Primary Axis (Left): Coverage --
    # Green shades: Darker for File 1, Lighter for File 2
    rects1 = ax1.bar(x - width/2, df1['Avg. Coverage per Head (Green)'], width, label='File 1 Coverage', color='#2E8B57') 
    rects2 = ax1.bar(x + width/2, df2['Avg. Coverage per Head (Green)'], width, label='File 2 Coverage', color='#66CDAA') 

    ax1.set_xlabel('Entity Degree Quantile Bin', fontsize=12)
    ax1.set_ylabel('Avg. Coverage per Head', color='green', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_ylim(0, 1.0) # Coverage is usually 0-1

    # -- Secondary Axis (Right): Head Frequency --
    # Using lines with markers to prevent visual clutter with the bars
    ax2 = ax1.twinx()
    line1 = ax2.plot(x, df1['Avg. Head Frequency (Orange)'], color='#FF8C00', label='File 1 Frequency', marker='o', linewidth=2)
    line2 = ax2.plot(x, df2['Avg. Head Frequency (Orange)'], color='#FFD700', label='File 2 Frequency', marker='s', linestyle='--', linewidth=2)

    ax2.set_ylabel('Avg. Head Frequency', color='#FF8C00', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#FF8C00')
    
    # Formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(bins, rotation=45, ha='right')
    plt.title('Comparison: Avg. Coverage and Head Frequency', fontsize=14)
    
    # Combined Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filename1)
    print(f"Saved {output_filename1}")


    # ==========================================
    # FIGURE 2: Candidates per Head (Grouped) + Head Frequency
    # ==========================================
    fig2, ax3 = plt.subplots(figsize=(18, 8))

    # -- Primary Axis (Left): Candidates --
    # Blue shades: Darker for File 1, Lighter for File 2
    rects3 = ax3.bar(x - width/2, df1['Avg. Candidates per Head (Blue)'], width, label='File 1 Candidates', color='#0000CD') 
    rects4 = ax3.bar(x + width/2, df2['Avg. Candidates per Head (Blue)'], width, label='File 2 Candidates', hatch='/', color='#4169E1') 

    ax3.set_xlabel('Entity Degree Quantile Bin', fontsize=12)
    ax3.set_ylabel('Avg. Candidates per Head', color='blue', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='blue')
    
    # -- Secondary Axis (Right): Head Frequency --
    ax4 = ax3.twinx()
    line3 = ax4.plot(x, df1['Avg. Head Frequency (Orange)'], color='#FF8C00', label='File 1 Frequency', marker='o', linewidth=2)
    line4 = ax4.plot(x, df2['Avg. Head Frequency (Orange)'], color='#FFD700', label='File 2 Frequency', marker='s', linestyle='--', linewidth=2)

    ax4.set_ylabel('Avg. Head Frequency', color='#FF8C00', fontsize=12)
    ax4.tick_params(axis='y', labelcolor='#FF8C00')

    # Formatting
    ax3.set_xticks(x)
    ax3.set_xticklabels(bins, rotation=45, ha='right')
    plt.title('Comparison: Avg. Candidates and Head Frequency', fontsize=14)
    
    # Combined Legend
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper left')

    ax3.grid(True, which='major', axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_filename2)
    print(f"Saved {output_filename2}")

# --- Instructions ---
# 1. Save your first CSV data as 'data_1.csv'
# 2. Save your second CSV data as 'data_2.csv'
# 3. Uncomment the line below to run:
p1 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/bcebceFB15k-237.csv"
p2 = "/home/juliansampels/Masterarbeit/MasterThesis/quantitativ/PPFB15k-237.csv"
plot_combined_metrics(p1, p2)