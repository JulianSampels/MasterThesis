import torch
import os
from PathE.pathe import data_utils as du

def analyze_entity_roles(datasets):
    for dataset in datasets:
        print(f"\nAnalyzing dataset: {dataset}")
        
        train_path = f'./data/path_datasets/{dataset}/train/'
        val_path = f'./data/path_datasets/{dataset}/val/'
        test_path = f'./data/path_datasets/{dataset}/test/'
        
        if not (os.path.exists(f'{train_path}/triples.pt') and os.path.exists(f'{val_path}/triples.pt') and os.path.exists(f'{test_path}/triples.pt')):
            print(f"  Triples not found for {dataset}, skipping.")
            continue
        
        # Load and filter triples for each split
        splits = ['train', 'val', 'test']
        split_triples = {}
        for split in splits:
            path = f'./data/path_datasets/{dataset}/{split}/'
            triples = torch.load(f'{path}/triples.pt')
            rel2inv = du.load_relation2inverse_relation_from_file(path)
            triples = triples[torch.isin(triples[:, 1], torch.tensor(list(rel2inv.keys()), dtype=torch.long, device=triples.device))]
            split_triples[split] = triples
        
        # Role analysis for test
        train_triples = split_triples['train']
        test_triples = split_triples['test']
        
        # Get unique entities in test
        test_entities = set(torch.unique(test_triples[:, [0, 2]]).tolist())
        total_test_entities = len(test_entities)
        
        # Filter train triples to only those where both head and tail are in test_entities
        test_entities_tensor = torch.tensor(list(test_entities), dtype=torch.long, device=train_triples.device)
        filtered_train_triples = train_triples[
            torch.isin(train_triples[:, 0], test_entities_tensor) & 
            torch.isin(train_triples[:, 2], test_entities_tensor)
        ]
        
        # Get heads and tails in filtered train
        train_heads = set(filtered_train_triples[:, 0].tolist())
        train_tails = set(filtered_train_triples[:, 2].tolist())
        
        # Get heads and tails in test
        test_heads = set(test_triples[:, 0].tolist())
        test_tails = set(test_triples[:, 2].tolist())
        
        # Counters
        both_in_train_only_head_in_test = 0
        both_in_train_only_tail_in_test = 0
        
        for ent in test_entities:
            # Check train roles
            in_train_head = ent in train_heads
            in_train_tail = ent in train_tails
            both_in_train = in_train_head and in_train_tail
            
            # Check test roles
            in_test_head = ent in test_heads
            in_test_tail = ent in test_tails
            
            if both_in_train:
                if in_test_head and not in_test_tail:
                    both_in_train_only_head_in_test += 1
                elif in_test_tail and not in_test_head:
                    both_in_train_only_tail_in_test += 1
        
        print(f"  Total entities in test: {total_test_entities}")
        print(f"  Entities that were both head/tail in train, now only head in test: {both_in_train_only_head_in_test}")
        print(f"  Entities that were both head/tail in train, now only tail in test: {both_in_train_only_tail_in_test}")
        combined = both_in_train_only_head_in_test + both_in_train_only_tail_in_test
        print(f"  Combined: entities that were both in train, now only one role in test: {combined}")
        if total_test_entities > 0:
            print(f"  Percentage only head: {both_in_train_only_head_in_test / total_test_entities * 100:.2f}%")
            print(f"  Percentage only tail: {both_in_train_only_tail_in_test / total_test_entities * 100:.2f}%")
            print(f"  Combined percentage: {combined / total_test_entities * 100:.2f}%")
        
        # Connectivity for all splits
        for split in splits:
            triples = split_triples[split]
            unique_heads = set(triples[:, 0].tolist())
            unique_tails = set(triples[:, 2].tolist())
            total_possible_pairs = len(unique_heads) * len(unique_tails)
            actual_pairs = set((row[0].item(), row[2].item()) for row in triples)
            num_actual_pairs = len(actual_pairs)
            connectivity_percentage = (num_actual_pairs / total_possible_pairs * 100) if total_possible_pairs > 0 else 0
            
            all_entities = unique_heads | unique_tails
            total_possible_overall = len(all_entities) ** 2
            overall_connectivity_percentage = (num_actual_pairs / total_possible_overall * 100) if total_possible_overall > 0 else 0
            
            print(f"  {split.capitalize()} Connectivity: {num_actual_pairs}/{total_possible_pairs} ({connectivity_percentage:.2f}%) head-tail, {num_actual_pairs}/{total_possible_overall} ({overall_connectivity_percentage:.4f}%) overall")

if __name__ == "__main__":
    datasets = ['codex-small', 'fb15k237', 'wn18rr', 'jf17k', 'jf17k_filtered']
    analyze_entity_roles(datasets)