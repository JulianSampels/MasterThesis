import torch
from PathE.pathe.figures import create_tail_occurrence_per_head_figure, create_relation_occurrence_figure, plot_relation_count_dispersion, plot_tail_count_dispersion
from PathE.pathe import data_utils as du
# Load the triples for codex-small train set

train_path = './data/path_datasets/codex-small/train/'
triples = torch.load(f'{train_path}/triples.pt')
train_rel2inv = du.load_relation2inverse_relation_from_file(train_path)
# Load relation to inverse relation mappings

# Filter out inverse relations from triples
triples = triples[torch.isin(triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=triples.device))]

# Generate the figures
create_tail_occurrence_per_head_figure(triples, save_dir="./figures", filename="tail_occurrence_per_head_codex_small.svg")
create_relation_occurrence_figure(triples, save_dir="./figures", filename="relation_occurrence_codex_small.svg")
plot_relation_count_dispersion(triples, save_dir="./figures", filename="relation_dispersion_codex_small.svg")
plot_tail_count_dispersion(triples, save_dir="./figures", filename="tail_dispersion_codex_small.svg")