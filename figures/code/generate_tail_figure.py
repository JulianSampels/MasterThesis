import torch
from PathE.pathe.figures import create_entity_occurrence_figure, create_relation_occurrence_figure, plot_entity_pair_multiplicity_dispersion, plot_relation_count_dispersion, plot_entity_degree_dispersion
from PathE.pathe import data_utils as du
# Load the triples for codex-small train set

# train_path = './data/path_datasets/codex-small/train/'
# train_path = './data/path_datasets/fb15k237/train/'
# train_path = './data/path_datasets/wn18rr/train/'
filename_end = "fb15k" 
filename_end = "codex-small" 
filename_end = "fb15k237"
filename_end = "wn18rr"
filename_end = "jf17k"
filename_end = "jf17k_filtered"
train_path = f'./data/path_datasets/{filename_end}/test/'

triples = torch.load(f'{train_path}/triples.pt')
train_rel2inv = du.load_relation2inverse_relation_from_file(train_path)
# Load relation to inverse relation mappings

# Filter out inverse relations from triples
triples = triples[torch.isin(triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=triples.device))]


# Generate the figures
plot_relation_count_dispersion(triples, save_dir=f"./figures/{filename_end}", filename=f"relation_dispersion_{filename_end}.svg")
plot_entity_pair_multiplicity_dispersion(triples, save_dir=f"./figures/{filename_end}", filename=f"entity_pair_dispersion_{filename_end}.svg")
# plot_entity_degree_dispersion(triples, save_dir=f"./figures/{filename_end}", filename=f"entity_dispersion_{filename_end}.svg")
# create_entity_occurrence_figure(triples, save_dir=f"./figures/{filename_end}", filename=f"entity_occurrence_{filename_end}.svg")
# create_relation_occurrence_figure(triples, save_dir=f"./figures/{filename_end}", filename=f"relation_occurrence_{filename_end}.svg")