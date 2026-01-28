import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from PathE.pathe import data_utils as du

# ----------------------------------------------------------------------------- #
# 1. Geometry & Font Configuration
# ----------------------------------------------------------------------------- #
TEXT_WIDTH_CM = 15.0  
TEXT_WIDTH_INCH = TEXT_WIDTH_CM / 2.54
FONT_SIZE_PT = 11

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", 
    "text.usetex": True,           
    "font.family": "sans-serif",
    
    "pgf.rcfonts": False,          
    
    "font.size": FONT_SIZE_PT,       
    "axes.labelsize": FONT_SIZE_PT,  
    "axes.titlesize": FONT_SIZE_PT + 1,
    "xtick.labelsize": FONT_SIZE_PT - 2, 
    "ytick.labelsize": FONT_SIZE_PT - 2,
    "legend.fontsize": FONT_SIZE_PT - 2, 
    
    "pgf.preamble": "\n".join([
         r"\usepackage[utf8]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{amsmath}",
         r"\usepackage{amssymb}",
         r"\usepackage{sansmath}",     
         r"\sansmath",                 
         r"\renewcommand{\familydefault}{\sfdefault}", 
         r"\providecommand{\mathdefault}[1]{#1}",
    ]),
    
    "figure.autolayout": False, 
})

# Datasets and display names
datasets = ["fb15k237", "jf17k", "wn18rr"]
display_names = {
    "fb15k237": "FB15k-237",
    "jf17k": "JF17k",
    "wn18rr": "WN18RR",
}
colors_datasets = ["#D21518", "#34A855", "#378DD4"]

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def count_hist(rows: torch.Tensor, cols: torch.Tensor, num_cols: int) -> Tuple[Dict[int, int], int]:
    unique_rows, inv = torch.unique(rows, return_inverse=True)
    order = torch.argsort(inv)
    rows_sorted = inv[order]
    cols_sorted = cols[order]
    hist: Dict[int, int] = defaultdict(int)
    max_val = 0
    n = rows_sorted.numel()
    start = 0
    for rid in range(len(unique_rows)):
        end = start
        while end < n and rows_sorted[end] == rid:
            end += 1
        row_cols = cols_sorted[start:end]
        if row_cols.numel() > 0:
            uniq_c, cnts = torch.unique(row_cols, return_counts=True)
            for c in cnts.tolist():
                hist[c] += 1
                if c > max_val:
                    max_val = c
            zeros = num_cols - uniq_c.numel()
        else:
            zeros = num_cols
        if zeros > 0:
            hist[0] += int(zeros)
        start = end
    return hist, max_val

def union_xs(hist_list: List[Dict[int, int]]) -> List[int]:
    xs = set()
    for h in hist_list:
        xs.update(h.keys())
    return sorted(xs)

def get_visual_padding(min_x, max_x, percentage=0.025):
    data_width = (max_x + 0.5) - (min_x - 0.5)
    pad = data_width * percentage
    new_min = (min_x - 0.5) - pad
    new_max = (max_x + 0.5) + pad
    return new_min, new_max

def plot_grouped_hist(ax, hist_list, title, xlabel, linewidth):
    xs = union_xs(hist_list)
    if not xs:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        return
    k = len(hist_list)
    bar_width = 0.8 / k
    offsets = np.linspace(-(k - 1) / 2, (k - 1) / 2, k) * bar_width
    centers = np.array(xs, dtype=float)
    for i, hist in enumerate(hist_list):
        ys = [hist.get(x, 0) for x in xs]
        ax.bar(
            centers + offsets[i], ys, width=bar_width, color=colors_datasets[i],
            alpha=1.0, label=display_names[datasets[i]], edgecolor="black",
            linewidth=linewidth, zorder=2,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Freq. (log)") 
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if xs:
        v_min, v_max = get_visual_padding(min(xs), max(xs))
        ax.set_xlim(v_min, v_max)

def plot_stepfilled_hist(ax, hist, title, xlabel, color, linewidth):
    if not hist: return
    xs = sorted(hist.keys())
    weights = [hist[x] for x in xs]
    min_x, max_x = xs[0], xs[-1]
    bins = np.arange(min_x - 0.5, max_x + 1.5, 1.0)
    ax.hist(
        xs, bins=bins, weights=weights, histtype='stepfilled', color=color,
        edgecolor='black', linewidth=linewidth, alpha=1.0, zorder=2
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Freq. (log)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    v_min, v_max = get_visual_padding(min_x, max_x)
    ax.set_xlim(v_min, v_max)

def plot_separated_bar(ax, hist, title, xlabel, color, linewidth):
    xs = sorted(hist.keys())
    ys = [hist[x] for x in xs]
    ax.bar(
        xs, ys, width=0.6, color=color, edgecolor="black",
        linewidth=linewidth, alpha=1.0, zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Freq. (log)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if xs: ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)

# ----------------------------------------------------------------------------- #
# Data prep
# ----------------------------------------------------------------------------- #
data = {}
for dataset in datasets:
    train_path = f"./data/path_datasets/{dataset}/test/"
    triples = torch.load(f"{train_path}/triples.pt")
    train_rel2inv = du.load_relation2inverse_relation_from_file(train_path)
    triples = triples[torch.isin(triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=triples.device))]
    triples = triples.cpu()
    heads = triples[:, 0]; tails = triples[:, 2]; rels = triples[:, 1]
    all_tails = torch.unique(tails); all_heads = torch.unique(heads); all_rels = torch.unique(rels)
    hist_tail_per_head, _ = count_hist(heads, tails, num_cols=len(all_tails))
    hist_head_per_tail, _ = count_hist(tails, heads, num_cols=len(all_heads))
    hist_rel_per_head, _ = count_hist(heads, rels, num_cols=len(all_rels))
    hist_rel_per_tail, _ = count_hist(tails, rels, num_cols=len(all_rels))
    data[dataset] = {
        "hist_tail_per_head": hist_tail_per_head, "hist_head_per_tail": hist_head_per_tail,
        "hist_rel_per_head": hist_rel_per_head, "hist_rel_per_tail": hist_rel_per_tail,
    }

hist_tail_combined = [data[d]["hist_tail_per_head"] for d in datasets]
hist_head_combined = [data[d]["hist_head_per_tail"] for d in datasets]

# ----------------------------------------------------------------------------- #
# Plotting
# ----------------------------------------------------------------------------- #
# Use 7.2 to be safe for vertical fit on page
HEIGHT_INCH = 7.2

fig, axes = plt.subplots(4, 2, figsize=(418.25555/72.27, HEIGHT_INCH), sharey="row")

# Row 0
plot_grouped_hist(axes[0, 0], hist_tail_combined, r"Tail Multiplicity per Head", "Relations per (Head, Tail) Pair", linewidth=0.7)
plot_grouped_hist(axes[0, 1], hist_head_combined, r"Head Multiplicity per Tail", "Relations per (Tail, Head) Pair", linewidth=0.7)
axes[0, 1].set_ylabel(""); axes[0, 1].legend(loc="upper right", frameon=True)

# Rows 1–3
for i, dataset in enumerate(datasets):
    row = i + 1; color = colors_datasets[i]; name = display_names[dataset]
    current_lw = 0.02 if row == 1 else 0.3
    if dataset == "jf17k":
        plot_separated_bar(axes[row, 0], data[dataset]["hist_rel_per_head"], f"{name}: Rel. Multiplicity per Head", "Tails per (Head, Relation) Pair", color, linewidth=0.7)
    else:
        plot_stepfilled_hist(axes[row, 0], data[dataset]["hist_rel_per_head"], f"{name}: Rel. Multiplicity per Head", "Tails per (Head, Relation) Pair", color, linewidth=current_lw)
    plot_stepfilled_hist(axes[row, 1], data[dataset]["hist_rel_per_tail"], f"{name}: Rel. Multiplicity per Tail", "Heads per (Tail, Relation) Pair", color, linewidth=current_lw)
    
    # -------------------------------------------------------
    # FIX: Reduce tick density specifically for FB15k-237 Tail plot
    # -------------------------------------------------------
    if dataset == "fb15k237":
        # Right column is axes[row, 1]
        axes[row, 1].xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))

    axes[row, 1].set_ylabel("")

fig.set_constrained_layout(True)
fig.get_layout_engine().set(h_pad=0.05, w_pad=0, hspace=0, wspace=0) 

os.makedirs("./figures", exist_ok=True)
output_path = "./figures/combined_occurrence_figure_test_split.pgf"

plt.savefig(output_path, format="pgf", bbox_inches='tight', pad_inches=0.01, transparent=False)
plt.close()

# ----------------------------------------------------------------------------- #
# POST-PROCESSING
# ----------------------------------------------------------------------------- #
print("Post-processing PGF...")
with open(output_path, "r", encoding="utf-8") as f:
    content = f.read()

injection = r"""
% Auto-injected by Python script
\providecommand{\mathdefault}[1]{#1}
\begingroup
\sffamily
\sansmath
"""

content = injection + content + r"\endgroup"

with open(output_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Figure saved and patched: {output_path}")