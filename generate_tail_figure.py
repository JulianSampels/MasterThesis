import torch
from PathE.pathe.figures import create_tail_occurrence_per_head_figure, create_relation_occurrence_figure, plot_relation_count_dispersion, plot_tail_count_dispersion
from PathE.pathe import data_utils as du
# Load the triples for codex-small train set

# train_path = './data/path_datasets/codex-small/train/'
# train_path = './data/path_datasets/fb15k237/train/'
train_path = './data/path_datasets/wn18rr/train/'
filename_end = "codex-small" 
filename_end = "fb15k237" 
filename_end = "wn18rr" 
filename_end = "fb15k" 
train_path = f'./data/path_datasets/{filename_end}/train/'

triples = torch.load(f'{train_path}/triples.pt')
train_rel2inv = du.load_relation2inverse_relation_from_file(train_path)
# Load relation to inverse relation mappings

# Filter out inverse relations from triples
triples = triples[torch.isin(triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=triples.device))]


# Generate the figures
create_tail_occurrence_per_head_figure(triples, save_dir="./figures", filename=f"tail_occurrence_per_head_{filename_end}.svg")
create_relation_occurrence_figure(triples, save_dir="./figures", filename=f"relation_occurrence_{filename_end}.svg")
plot_relation_count_dispersion(triples, save_dir="./figures", filename=f"relation_dispersion_{filename_end}.svg")
plot_tail_count_dispersion(triples, save_dir="./figures", filename=f"tail_dispersion_{filename_end}.svg")