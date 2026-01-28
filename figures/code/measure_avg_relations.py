import torch
import os
from PathE.pathe import data_utils as du
from collections import defaultdict

def measure_avg_relations_per_entity(dataset, split):
    path = f'./data/path_datasets/{dataset}/{split}/'
    if not os.path.exists(f'{path}/triples.pt'):
        print(f"Triples not found for {dataset} {split}")
        return
    
    triples = torch.load(f'{path}/triples.pt')
    rel2inv = du.load_relation2inverse_relation_from_file(path)
    # Filter to original relations (not inverses)
    original_rels = set(rel2inv.keys())
    triples = triples[torch.isin(triples[:, 1], torch.tensor(list(original_rels), dtype=torch.long, device=triples.device))]
    
    # Collect unique relations per entity
    rel_per_entity = defaultdict(set)
    for triple in triples:
        h, r, t = triple.tolist()
        rel_per_entity[h].add(r)
        rel_per_entity[t].add(r)
    
    # Calculate average unique relations per entity
    counts = [len(rels) for rels in rel_per_entity.values()]
    avg_relations = sum(counts) / len(counts) if counts else 0
    num_entities = len(rel_per_entity)
    
    print(f"Measuring average relations per entity in {dataset} {split}")
    print(f"  Entities: {num_entities}, Average unique relations per entity: {avg_relations:.2f}")

if __name__ == "__main__":
    datasets = ['codex-small', 'fb15k237', 'wn18rr', 'jf17k', 'jf17k_filtered']
    splits = ['train', 'val', 'test']
    for dataset in datasets:
        for split in splits:
            measure_avg_relations_per_entity(dataset, split)
        print("\n")