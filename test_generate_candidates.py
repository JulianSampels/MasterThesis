import torch
import os
import sys
sys.path.append('/home/juliansampels/Masterarbeit/MasterThesis/PathE')

from PathE.pathe.data_utils import load_tuple_tensors, load_triple_tensors, load_relation2inverse_relation_from_file
from PathE.pathe.candidates import CandidateGeneratorGlobalWithTail, grid_search_candidate_sizes, grid_search_candidates
from PathE.pathe.pathdata import RelationMaps
from PathE.pathe import triple_lib
from argparse import Namespace

# Load data
train_path = './data/path_datasets/codex-small/train/'
val_path = './data/path_datasets/codex-small/val/'
test_path = './data/path_datasets/codex-small/test/'

train_tuples, val_tuples, test_tuples = load_tuple_tensors(train_path, val_path, test_path)
train_triples, val_triples, test_triples = load_triple_tensors(train_path, val_path, test_path)

# Load relation to inverse relation mappings
train_rel2inv = load_relation2inverse_relation_from_file(train_path)
val_rel2inv = load_relation2inverse_relation_from_file(val_path)
test_rel2inv = load_relation2inverse_relation_from_file(test_path)

# Get unique entities and relations
unique_entities = triple_lib.get_unique_entities(train_triples, val_triples, test_triples)
num_entities = unique_entities.size(0)
unique_relations = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
num_relations = unique_relations.size(0)

print(f"Num entities: {num_entities}, Num relations: {num_relations}")
print(f"Train tuples: {train_tuples.shape}, Train triples: {train_triples.shape}")

# Precompute counts for efficiency
def compute_counts(triples, num_entities, num_relations):
    relation_counts = torch.zeros(num_entities, num_relations, dtype=torch.float)
    tail_counts = torch.zeros(num_entities, num_entities, dtype=torch.float)
    for h, r, t in triples:
        relation_counts[h, r] += 1
        tail_counts[h, t] += 1
    return relation_counts, tail_counts

train_relation_counts, train_tail_counts = compute_counts(train_triples, num_entities, num_relations)
val_relation_counts, val_tail_counts = compute_counts(val_triples, num_entities, num_relations)
test_relation_counts, test_tail_counts = compute_counts(test_triples, num_entities, num_relations)

# The candidate generator expects logits per tuple and aggregates them per head.
# To simulate perfect per-head predictions, we create inputs where each head appears once.
unique_heads_train = torch.unique(train_triples[:, 0])
unique_heads_val = torch.unique(val_triples[:, 0])
unique_heads_test = torch.unique(test_triples[:, 0])

# Create tuples input: (head, dummy_relation=0) for each unique head
train_tuples_all = torch.stack([unique_heads_train, torch.zeros_like(unique_heads_train)], dim=1)
va_tuples_all = torch.stack([unique_heads_val, torch.zeros_like(unique_heads_val)], dim=1)
test_tuples_all = torch.stack([unique_heads_test, torch.zeros_like(unique_heads_test)], dim=1)

train_tuples_all = torch.unique(train_tuples_all, dim=0)
va_tuples_all = torch.unique(va_tuples_all, dim=0)
test_tuples_all = torch.unique(test_tuples_all, dim=0)

# Create logits_rp input: (num_unique_heads, num_relations) with log(count(h,r))
train_logits_all = torch.log(train_relation_counts[unique_heads_train, :])
va_logits_all = torch.log(val_relation_counts[unique_heads_val, :])
test_logits_all = torch.log(test_relation_counts[unique_heads_test, :])

# Create logits_tp input: (num_unique_heads, num_entities) with log(count(h,t))
train_logits_tp_all = torch.log(train_tail_counts[unique_heads_train, :])
va_logits_tp_all = torch.log(val_tail_counts[unique_heads_val, :])
test_logits_tp_all = torch.log(test_tail_counts[unique_heads_test, :])

# For the additional test (train counts on test tuples)
test_logits_all_train_counts = torch.log(train_relation_counts[unique_heads_test, :])
test_logits_tp_all_train_counts = torch.log(train_tail_counts[unique_heads_test, :])

print("Computed logits per unique head")

# Create relation maps (using train rel2inv)
relation_maps = RelationMaps(
    original_relation_to_inverse_relation=train_rel2inv
)

# Create dummy datasets (minimal)
class DummyDataset:
    def __init__(self, relation_maps):
        self.relation_maps = relation_maps

train_set_t = DummyDataset(relation_maps)
valid_set_t = DummyDataset(relation_maps)
test_set_t = DummyDataset(relation_maps)

# Create args
args = Namespace(
    num_workers=16,
    figure_dir='./figures/test/testtuples',
    group_strategy=0,  # head
    candidates_threshold_p=None,
    candidates_quantile_q=None,
    candidates_temperature=1.0,
    candidates_alpha=0.5,
    candidates_beta=0.5,
    candidates_cap=100,
    candidates_normalize_mode='global_joint',
    phase1_loss_fn='poisson'
)

# Create candidate generator
candidate_generator = CandidateGeneratorGlobalWithTail(
    p=args.candidates_threshold_p,
    q=args.candidates_quantile_q,
    temperature=args.candidates_temperature,
    alpha=args.candidates_alpha,
    beta=args.candidates_beta,
    per_group_cap=args.candidates_cap,
    normalize_mode=args.candidates_normalize_mode,
    max_num_workers=args.num_workers,
    phase1_loss_fn=args.phase1_loss_fn
)


# Filter out inverse relations from triples
train_triples = train_triples[torch.isin(train_triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=train_triples.device))]
val_triples   = val_triples[torch.isin(val_triples[:, 1], torch.tensor(list(val_rel2inv.keys()), dtype=torch.long, device=val_triples.device))]
test_triples  = test_triples[torch.isin(test_triples[:, 1], torch.tensor(list(test_rel2inv.keys()), dtype=torch.long, device=test_triples.device))]


# update args for figure dir
args.figure_dir = './figures/test/traincountsontesttriples'
# Additional test: grid search with train counts on test tuples
print("Running grid search with train counts on test tuples...")
grid_search_candidates(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    test_tuples_all, test_logits_all_train_counts, test_logits_tp_all_train_counts,
    train_triples, val_triples, test_triples,
    train_set_t, valid_set_t, test_set_t
)
grid_search_candidate_sizes(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    test_tuples_all, test_logits_all_train_counts, test_logits_tp_all_train_counts,
    train_triples, val_triples, test_triples,
    train_set_t, valid_set_t, test_set_t
)

# Run grid search
# train
# update args for figure dir
args.figure_dir = './figures/test/traintuples'
grid_search_candidates(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    train_triples, val_triples, train_triples,
    train_set_t, valid_set_t, test_set_t
)
grid_search_candidate_sizes(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    train_triples, val_triples, train_triples,
    train_set_t, valid_set_t, test_set_t
)

#test
# update args for figure dir
args.figure_dir = './figures/test/testtuples'
grid_search_candidates(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    test_tuples_all, test_logits_all, test_logits_tp_all,
    train_triples, val_triples, test_triples,
    train_set_t, valid_set_t, test_set_t
)
grid_search_candidate_sizes(
    candidate_generator,
    args,
    train_tuples_all, train_logits_all, train_logits_tp_all,
    va_tuples_all, va_logits_all, va_logits_tp_all,
    test_tuples_all, test_logits_all, test_logits_tp_all,
    train_triples, val_triples, test_triples,
    train_set_t, valid_set_t, test_set_t
)

print("Done")