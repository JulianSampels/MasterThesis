from PathE.pathe.kgloader import KgLoader
from PathE.pathe.pathdataset import PathDataset

# dataset_name = 'fb15k237'
dataset_name = 'codex-small'
# dataset_name = 'codex-medium'
# dataset_name = 'codex-large'
# dataset_name = 'wn18rr'
# dataset_name = 'yago'
# dataset_name = 'wiki5m'
# dataset_name = "test-dataset"
# dataset_name = 'ogb-wikikg2'

# # 1. Load and split the dataset
dataset = KgLoader(dataset_name, add_inverse=True)
# print(f'train triples count: {len(dataset.train_triples)}')
# print(f'train tuples count: {len(dataset.train_tuples)}')

# # 2. Generate paths and save triples.pt for each split
path_handler = PathDataset(dataset=dataset, num_paths_per_entity=20, num_steps=10, parallel=True)
print(f'datasets created successfully in directory {path_handler.dataset_dir}.')