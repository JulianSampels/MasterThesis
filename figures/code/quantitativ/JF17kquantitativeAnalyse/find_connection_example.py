import os
import json

def load_entity_types(file_path):
    types = {}
    with open(file_path, 'r') as f:
        for line in f:
            ent, typ = line.strip().split('\t')
            types.setdefault(ent, set()).add(typ)
    return types

def load_triples(file_path):
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            h, t, r = line.strip().split()
            triples.append((h, r, t))
    return triples

def find_examples(train_file, test_file, types_file, train_path, num_examples=5):
    entity_types = load_entity_types(types_file)
    train_triples = load_triples(train_file)
    test_triples = load_triples(test_file)

    # Load mapping and keep only non-inverse relations (keys)
    with open(os.path.join(train_path, 'relation2inverseRelation.json'), 'r') as f:
        rel2inv = json.load(f)
    with open(os.path.join(train_path, 'id2relation.json'), 'r') as f:
        id2rel = json.load(f)
    valid_rels = {id2rel[rid] for rid in rel2inv if rid in id2rel}

    # Precompute train pairs
    train_hr = set()
    train_rt = set()
    train_ht = set()
    head_tails = {}
    head_types = {}
    for h, r, t in train_triples:
        if r not in valid_rels:
            continue
        train_hr.add((h, r))
        train_rt.add((r, t))
        train_ht.add((h, t))
        head_tails.setdefault(h, set()).add(t)
        if t in entity_types:
            head_types.setdefault(h, set()).update(entity_types[t])

    # Heads present in both splits
    common_heads = {h for h, r, t in train_triples if r in valid_rels} & \
                   {h for h, r, t in test_triples if r in valid_rels}

    forbidden_substrings = ["award", "popstra", "landcover", "people", "award", "event"]

    examples = []
    for h in sorted(common_heads):
        if len(head_tails.get(h, ())) < 1:
            continue
        for th, tr, tt in test_triples:
            if th != h or tr not in valid_rels:
                continue
            # Enforce all three exclusions
            if (h, tr) in train_hr:
                continue            # (h, r, *) seen
            if (tr, tt) in train_rt:
                continue            # (*, r, t) seen
            if (h, tt) in train_ht:
                continue            # (h, *, t) seen
            # Need overlapping types with some train tail of h
            if tt not in entity_types:
                continue
            matching_types = entity_types[tt] & head_types.get(h, set())
            if not matching_types:
                continue
            # Exclude if any matching type contains forbidden substrings
            if any(any(sub in typ for sub in forbidden_substrings) for typ in matching_types):
                continue
            
            examples.append({
                'head': h,
                'train_tails': sorted(head_tails[h]),
                'train_tail_types': sorted(head_types.get(h, set())),
                'test_triple': (th, tr, tt),
                'tail_types': sorted(entity_types[tt]),
                'matching_types': sorted(matching_types)
            })
            if len(examples) >= num_examples:
                break
        if len(examples) >= num_examples:
            break

    for i, ex in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Head: {ex['head']}")
        print(f"Train tails: {ex['train_tails']}")
        print(f"Train tail types: {ex['train_tail_types']}")
        print(f"Test triple: {ex['test_triple']}")
        print(f"Tail types: {ex['tail_types']}")
        print(f"Matching types: {ex['matching_types']}")
        print("-" * 50)

if __name__ == "__main__":
    train_file = "train.txt"
    test_file = "test.txt"
    types_file = "entity2types_ttv.txt"
    train_path = "../data/path_datasets/jf17k/train/"
    find_examples(train_file, test_file, types_file, train_path, num_examples=150)