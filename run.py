from PathE.pathe.kgloader import KgLoader
from PathE.pathe.pathdataset import PathDataset

# # 1. Load and split the dataset
# kg = KgLoader(dataset='fb15k237', add_inverse=True)

# # 2. Generate paths and save triples.pt for each split
# for part in ['train', 'valid', 'test']:
#     pd = PathDataset(kg, num_paths_per_entity=50, num_steps=20)
#     pd.make_random_walk_dataset(num_paths=50, num_steps=20)
#     pd.save_triple_tensor_and_dicts(part)
#     pd.save_relational_context(part)

# dataset_name = 'fb15k237'
dataset_name = 'codex-small'
# dataset_name = 'codex-medium'
# dataset_name = 'codex-large'
# dataset_name = 'wn18rr'
# dataset_name = 'yago'
# dataset_name = 'wiki5m'
# dataset_name = "test-dataset"
# dataset_name = 'ogb-wikikg2'
dataset = KgLoader(dataset_name, add_inverse=True)
# print(f'train triples count: {len(dataset.train_triples)}')
# print(f'train tuples count: {len(dataset.train_tuples)}')
path_handler = PathDataset(dataset=dataset, num_paths_per_entity=20, num_steps=10, parallel=True)
print(f'datasets created successfuly in directory {path_handler.dataset_dir}.')