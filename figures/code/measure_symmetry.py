import torch
import os
from PathE.pathe import data_utils as du
from collections import defaultdict

def measure_symmetric_relations(dataset='wn18rr', split='train'):
    if split == 'full':
        # Combine all splits
        all_triples = []
        for s in ['train', 'val', 'test']:
            path = f'./data/path_datasets/{dataset}/{s}/'
            if os.path.exists(f'{path}/triples.pt'):
                triples = torch.load(f'{path}/triples.pt')
                all_triples.append(triples)
        if not all_triples:
            print(f"No triples found for {dataset} full")
            return
        triples = torch.cat(all_triples, dim=0)
    else:
        path = f'./data/path_datasets/{dataset}/{split}/'
        if not os.path.exists(f'{path}/triples.pt'):
            print(f"Triples not found for {dataset} {split}")
            return
        triples = torch.load(f'{path}/triples.pt')
    
    rel2inv = du.load_relation2inverse_relation_from_file(f'./data/path_datasets/{dataset}/train/')  # Use train for rel2inv
    # Filter to original relations (not inverses)
    original_rels = set(rel2inv.keys())
    triples = triples[torch.isin(triples[:, 1], torch.tensor(list(original_rels), dtype=torch.long, device=triples.device))]
    
    # Group triples by relation
    rel_to_triples = defaultdict(list)
    for triple in triples:
        h, r, t = triple.tolist()
        rel_to_triples[r].append((h, t))
    
    print(f"Measuring symmetric relations in {dataset} {split}")
    total_symmetric_triples = 0
    total_triples = 0
    
    for r, pairs in rel_to_triples.items():
        pair_set = set(pairs)
        symmetric_count = 0
        for h, t in pairs:
            if (t, h) in pair_set:
                symmetric_count += 1
        percentage = (symmetric_count / len(pairs) * 100) if pairs else 0
        # print(f"  Relation {r}: {symmetric_count}/{len(pairs)} ({percentage:.2f}%) symmetric triples")
        total_symmetric_triples += symmetric_count
        total_triples += len(pairs)
    
    overall_percentage = (total_symmetric_triples / total_triples * 100) if total_triples else 0
    print(f"Overall: {total_symmetric_triples}/{total_triples} ({overall_percentage:.2f}%) symmetric triples")
    

    # Analyze symmetric entity pairs (across all relations)
    all_pairs = set((h, t) for r, pairs in rel_to_triples.items() for (h, t) in pairs)
    symmetric_triples_count = 0
    for r, pairs in rel_to_triples.items():
        for h, t in pairs:
            if (t, h) in all_pairs:
                symmetric_triples_count += 1
    pair_percentage = (symmetric_triples_count / total_triples * 100) if total_triples else 0
    print(f"Symmetric entity pairs: {symmetric_triples_count}/{total_triples} ({pair_percentage:.2f}%) of triples have symmetric (t,h) pair")

if __name__ == "__main__":
    datasets = ['codex-small', 'fb15k237', 'wn18rr', 'jf17k', 'jf17k_filtered']
    splits = ['train', 'val', 'test', 'full']
    for dataset in datasets:
        for split in splits:
            measure_symmetric_relations(dataset, split)
        print("\n")