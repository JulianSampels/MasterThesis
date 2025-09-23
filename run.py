from PathE.pathe.kgloader import KgLoader
from PathE.pathe.pathdataset import PathDataset

dataset_name = 'fb15k237'
# dataset_name = 'codex-small'
# dataset_name = 'codex-medium'
# dataset_name = 'codex-large'
# dataset_name = 'wn18rr'
# dataset_name = 'yago'
# dataset_name = 'wiki5m'
# dataset_name = "test-dataset"
# dataset_name = 'ogb-wikikg2'

# # 1. Load and split the dataset
dataset = KgLoader(dataset_name, automatically_add_inverse=False, manually_add_inverse=True)
# print(f'train triples count: {len(dataset.train_triples)}')
# print(f'train tuples count: {len(dataset.train_tuples)}')
# print(f'train relation count: {len(dataset.train_triples[:, 1].unique())}')
# print(f'val triples count: {len(dataset.val_triples)}')
# print(f'val tuples count: {len(dataset.val_tuples)}')
# print(f'val relation count: {len(dataset.val_triples[:, 1].unique())}')
# print(f'test triples count: {len(dataset.test_triples)}')
# print(f'test tuples count: {len(dataset.test_tuples)}')
# print(f'test relation count: {len(dataset.test_triples[:, 1].unique())}')

# # 2. Generate paths and save triples.pt for each split
path_handler = PathDataset(dataset=dataset, num_paths_per_entity=20, num_steps=10, parallel=True)
print(f'datasets created successfully in directory {path_handler.dataset_dir}.')