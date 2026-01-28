"""
Scalable routines for training the PathE model within the Split-Join-Predict Framework.

Reference: Chapter 5: A Framework for Edge-Agnostic Completion
"""
import copy
import os
import logging
import datetime
import gc
from functools import partial
from typing import Tuple

import torch
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

# Optimisation for Tensor Cores (Section 5.4: Implementation and Scalability Optimisations)
torch.set_float32_matmul_precision('high') 

from .candidates import (
    CandidateGeneratorGlobal, 
    CandidateGeneratorGlobalWithTail, 
    CandidateGeneratorPerHead,
    grid_search_candidates,
    grid_search_candidate_sizes
)
from .pathe_ranking_metrics import get_metric_mode, evaluate_candidate_baseline_metrics
from .pather_models import PathEModelTriples, PathEModelTuplesMultiHead
from .pathdata import (
    CandidateTripleEntityMultiPathDataset, 
    UniqueHeadEntityMultiPathDataset, 
    create_vocabulary_from_relations
)
from .data_utils import (
    collate_multipaths, 
    load_triple_tensors, 
    load_tuple_tensors
)
from .callbacks import DatasetUpdater
from .utils import stageprint, bundle_arguments, namespace_to_dict
from .wrappers import (
    PathEModelWrapperTriples, 
    PathEModelWrapperUniqueHeads
)
from .path_lib import encode_relcontext_freqs
from .figures import create_figures
from . import data_utils as du
from . import triple_lib

logger = logging.getLogger(__name__)

# Global max workers to avoid excessive spawning during prediction
MAX_WORKERS_PREDICTION = 16
MAX_WORKERS_TEST = 16


def shutdown_dataloader(loader: torch.utils.data.DataLoader) -> None:
    """Aggressively stop persistent DataLoader workers to free memory."""
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
        logger.info("Successfully shutdown dataloader workers.")
    except Exception as e:
        logger.warning(f"Could not shutdown dataloader workers: {e}")


def predict_all(
    trainer: Trainer, 
    model: PathEModelWrapperUniqueHeads, 
    loader: torch.utils.data.DataLoader, 
    ckpt_path: str = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes inference over the entire dataset to generate property predictions.
    
    This enables the decoupled inference described in Section 5.1, allowing 
    property scores to be computed once and reused for candidate generation.
    """
    # Create a transient dataloader without persistent workers for one-off inference
    pred_loader = torch.utils.data.DataLoader(
        loader.dataset, 
        batch_size=loader.batch_size, 
        collate_fn=loader.collate_fn, 
        shuffle=False,
        pin_memory=loader.pin_memory, 
        num_workers=min(loader.num_workers, MAX_WORKERS_PREDICTION),
        persistent_workers=False 
    )
    
    outs = trainer.predict(model, dataloaders=pred_loader, ckpt_path=ckpt_path)

    # Aggregate batches
    tuples_all = torch.cat([o["tuples"].cpu() for o in outs], dim=0)
    scores_rp_all = torch.cat([o["scores_rp"].cpu() for o in outs], dim=0)
    scores_tp_all = torch.cat([o["scores_tp"].cpu() for o in outs], dim=0)
    
    del outs, pred_loader
    return tuples_all, scores_rp_all, scores_tp_all


def run_phase_1_property_prediction(
    args, 
    train_tuples, val_tuples, test_tuples,
    train_triples, val_triples, test_triples,
    paths, relcon, tokens_to_idxs,
    train_rel2inv, val_rel2inv, test_rel2inv
):
    """
    Executes Phase 1: Property Prediction (Section 5.3.1).
    
    Implements Entity-Centric prediction using the Unique-Head training paradigm.
    Optimises the Binary Existence or Frequency Estimation objectives.
    """
    stageprint(f"Phase 1: Property Prediction ({args.expname})")

    # 1. Pre-compute sparse adjacency matrices (Section 5.4.1)
    # This avoids repetitive host-to-device transfers.
    num_entities = triple_lib.get_unique_entities(train_triples, val_triples, test_triples).size(0)
    num_relations = len(tokens_to_idxs) - 2 # Exclude PAD/MSK
    
    train_head_tail_adj = triple_lib.get_full_adjacency_matrix(train_triples, num_entities)
    val_head_tail_adj = triple_lib.get_full_adjacency_matrix(val_triples, num_entities)
    test_head_tail_adj = triple_lib.get_full_adjacency_matrix(test_triples, num_entities)

    train_rel_count_mat = triple_lib.get_relation_count_matrix(train_tuples, num_entities, num_relations)
    val_rel_count_mat = triple_lib.get_relation_count_matrix(val_tuples, num_entities, num_relations)
    test_rel_count_mat = triple_lib.get_relation_count_matrix(test_tuples, num_entities, num_relations)

    # 2. Dataset Initialization with Path-Based Data Augmentation
    # Augmentation factor helps recover sample richness lost in entity-centric view.
    parallel = True
    train_set_t = UniqueHeadEntityMultiPathDataset(
        path_store=paths, relcontext_store=relcon,
        tuple_store=train_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=train_rel2inv,
        maximum_tuple_paths=args.max_ppt,
        parallel=parallel, num_workers=args.num_workers,
        head_tail_adjacency=train_head_tail_adj,
        tokens_to_idxs=tokens_to_idxs,
        augmentation_factor=args.augmentation_factor 
    )
    
    # Share memory for valid/test to reduce overhead
    shared_path_store = (train_set_t.relation_paths, train_set_t.entity_paths, train_set_t.path_index)
    
    valid_set_t = UniqueHeadEntityMultiPathDataset(
        path_store=shared_path_store, relcontext_store=relcon,
        tuple_store=val_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=val_rel2inv,
        maximum_tuple_paths=args.max_ppt, tokens_to_idxs=tokens_to_idxs,
        parallel=parallel, num_workers=args.num_workers,
        head_tail_adjacency=val_head_tail_adj,
        augmentation_factor=1
    )
    test_set_t = UniqueHeadEntityMultiPathDataset(
        path_store=shared_path_store, relcontext_store=relcon,
        tuple_store=test_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=test_rel2inv,
        maximum_tuple_paths=args.max_ppt, tokens_to_idxs=tokens_to_idxs,
        parallel=parallel, num_workers=args.num_workers,
        head_tail_adjacency=test_head_tail_adj,
        augmentation_factor=1
    )

    # 3. DataLoader Optimisation
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    use_cuda = (args.device == "cuda")
    use_persist = args.num_workers > 0
    
    tr_loader = torch.utils.data.DataLoader(
        train_set_t, batch_size=args.batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist
    )
    va_loader = torch.utils.data.DataLoader(
        valid_set_t, batch_size=args.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist
    )
    te_loader = torch.utils.data.DataLoader(
        test_set_t, batch_size=args.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=min(args.num_workers, MAX_WORKERS_TEST),
        persistent_workers=False
    )

    # 4. Model Architecture: Prediction Heads
    # Select Multi-Head if using complex loss functions (e.g. Negative Binomial)
    multi_head_losses = ['poisson', 'hurdle', 'negative_binomial']
    model_class = PathEModelTuplesMultiHead
    
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon, num_entities=num_entities,
        num_relations=train_set_t.vocab_size - 2, offset=2
    )
    
    bundle = partial(bundle_arguments, exclude=["vocab_size"], args=namespace_to_dict(args))
    model_t = model_class(
        vocab_size=train_set_t.vocab_size,
        relcontext_graph=relcontext_graph,
        padding_idx=tokens_to_idxs["PAD"],
        relation_multi_head=(args.phase1_rp_loss_fn in multi_head_losses),
        tail_multi_head=(args.phase1_tp_loss_fn in multi_head_losses),
        **bundle(target_class=model_class),
    )

    # Wrapper handles Multi-Task Learning (Section 5.4.1)
    map_head_to_relsets = triple_lib.make_relation_filter_dict_no_sp_tokens_tuples(train_tuples, val_tuples, test_tuples)
    class_weights = triple_lib.get_class_weights_without_special_tokens(train_triples) if args.class_weights else None

    pl_model_t = PathEModelWrapperUniqueHeads(
        pathe_model=model_t,
        filtration_dict=map_head_to_relsets,
        train_head_tail_adjacency=train_head_tail_adj,
        val_head_tail_adjacency=val_head_tail_adj,
        test_head_tail_adjacency=test_head_tail_adj,
        train_relation_count_matrix=train_rel_count_mat,
        val_relation_count_matrix=val_rel_count_mat,
        test_relation_count_matrix=test_rel_count_mat,
        class_weights=class_weights,
        **namespace_to_dict(args),
    )
    pl_model_t.model.to(torch.device(args.device))

    # 5. Trainer Setup
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=args.expname, version=args.version, sub_dir="phase1_properties")
    wb_logger = None
    if args.wandb_project is not None:
        wb_logger = WandbLogger(id=args.wandb_id, save_dir=tb_logger.log_dir, name=f"{args.expname}_phase1",
                                project=args.wandb_project, log_model="all", sync_tensorboard=True)
        wb_logger.watch(pl_model_t, log="all")

    checkpoint_callback = ModelCheckpoint(
        monitor=args.tuple_monitor, dirpath=args.checkpoint_dir, 
        mode=get_metric_mode(pl_model_t, args.tuple_monitor),
        filename=args.model + f"-phase1-{{epoch}}-{{{args.tuple_monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps
    )
    
    # Manual Optimization (Section 5.4.1)
    # Necessary to control gradient flow for both prediction heads.
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.device == "cuda" else "cpu", 
        devices=args.num_devices, 
        limit_train_batches=(0.1 if args.debug else 1.0),
        logger=[tb_logger] + ([wb_logger] if wb_logger else []),
        val_check_interval=args.val_check_interval,
        callbacks=[
            EarlyStopping(monitor=args.tuple_monitor, patience=args.patience, mode=get_metric_mode(pl_model_t, args.tuple_monitor)),
            checkpoint_callback, 
            DatasetUpdater([train_set_t, valid_set_t] if args.train_paths else [train_set_t.dataset])
        ],
        gradient_clip_val=1.0 if not args.use_manual_optimization else None, 
        accumulate_grad_batches=args.accumulate_gradient if not args.use_manual_optimization else 1
    )

    if args.cmd in ["train", "resume"] and not args.skip_phase1:
        trainer.fit(pl_model_t, tr_loader, va_loader, ckpt_path=args.tuple_checkpoint)
        tuple_ckpt = checkpoint_callback.best_model_path
        logger.info(f"[Phase 1] Training complete. Best model: {tuple_ckpt}")
    else:
        tuple_ckpt = args.tuple_checkpoint
        logger.info(f"[Phase 1] Skipping training. Using checkpoint: {tuple_ckpt}")
    
    # 6. Evaluate the tuple model if requested
    if not args.skip_phase1:
        stageprint("Evaluating Phase 1 Tuple Model (Relation Predictor) on Test Set...")
        tuple_test_dict = trainer.test(
            model=pl_model_t if args.cmd == "test" else None,
            dataloaders=te_loader,
            ckpt_path=tuple_ckpt
        )[0]
        logger.info(f"Phase 1 Testing results: {tuple_test_dict}")
        pd.DataFrame([tuple_test_dict]).to_csv(os.path.join(tb_logger.log_dir, "phase1_results_summary.csv"), index=False)

    # Return artifacts needed for Phase 2
    return trainer, pl_model_t, tuple_ckpt, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t


def run_phase_2_candidate_generation(
    args, trainer, pl_model, ckpt_path, 
    tr_loader, va_loader, te_loader,
    train_set_t, valid_set_t, test_set_t,
    train_triples, val_triples, test_triples
):
    """
    Executes Phase 2: Candidate Generation (Section 5.3.2).
    
    Joins decoupled scores into a probabilistic joint score S(h,r,t).
    Performs global Top-k selection to filter the search space.
    """
    stageprint(f"Phase 2: Candidate Generation (Global Join & Top-k Filtering)")
    
    # 1. Inference: Predict property scores for all tuples
    stageprint("...Generating property scores for all splits...")
    tr_tuples, tr_scores_rp, tr_scores_tp = predict_all(trainer, pl_model, tr_loader, ckpt_path=ckpt_path)
    va_tuples, va_scores_rp, va_scores_tp = predict_all(trainer, pl_model, va_loader, ckpt_path=ckpt_path)
    te_tuples, te_scores_rp, te_scores_tp = predict_all(trainer, pl_model, te_loader, ckpt_path=ckpt_path)

    # Clean up Phase 1 resources immediately to avoid OOM
    shutdown_dataloader(tr_loader)
    shutdown_dataloader(va_loader)
    shutdown_dataloader(te_loader)
    del tr_loader, va_loader, te_loader, trainer, pl_model
    gc.collect()
    if args.device == "cuda": torch.cuda.empty_cache()

    # 2. Select Candidate Generator Strategy
    if args.candidate_generator == 'global':
        cand_gen = CandidateGeneratorGlobal(
            p=args.candidates_threshold_p, q=args.candidates_quantile_q, 
            temperature=args.candidates_temperature, alpha=args.candidates_alpha, 
            per_group_cap=args.candidates_cap, normalize_mode=args.candidates_normalize_mode, 
            max_num_workers=args.num_workers
        )
    elif args.candidate_generator == 'global_with_tail':
        cand_gen = CandidateGeneratorGlobalWithTail(
            p=args.candidates_threshold_p, q=args.candidates_quantile_q, 
            temperature=args.candidates_temperature, alpha=args.candidates_alpha, 
            beta=args.candidates_beta, per_group_cap=args.candidates_cap, 
            normalize_mode=args.candidates_normalize_mode, max_num_workers=args.num_workers
        )
    elif args.candidate_generator == 'per_head':
        cand_gen = CandidateGeneratorPerHead(
            per_group_cap=args.candidates_cap, 
            alpha=args.candidates_alpha
        )
    else:
        raise ValueError(f"Unknown candidate_generator: {args.candidate_generator}")

    # 3. Grid Search for Hyperparameters alpha/beta (Eq. 5.4)
    if True: 
        grid_search_candidates(
            cand_gen, args, 
            tr_tuples, tr_scores_rp, tr_scores_tp, 
            va_tuples, va_scores_rp, va_scores_tp, 
            te_tuples, te_scores_rp, te_scores_tp, 
            train_triples, val_triples, test_triples, 
            train_set_t, valid_set_t, test_set_t
        )
    
    # 3b. Grid Search over candidate sizes
    if True:
        grid_search_candidate_sizes(
            cand_gen, args, 
            tr_tuples, tr_scores_rp, tr_scores_tp, 
            va_tuples, va_scores_rp, va_scores_tp, 
            te_tuples, te_scores_rp, te_scores_tp, 
            train_triples, val_triples, test_triples, 
            train_set_t, valid_set_t, test_set_t
        )

    # 4. Generate Candidates
    # Note: Search space is decomposed by relation partitions for memory efficiency
    num_groups_train = len(torch.unique(train_triples[:, args.group_strategy], dim=0))
    num_groups_val = len(torch.unique(val_triples[:, args.group_strategy], dim=0))
    num_groups_test = len(torch.unique(test_triples[:, args.group_strategy], dim=0))

    candidates_train, _ = cand_gen.generate_candidates(tr_tuples, tr_scores_rp, train_set_t.relation_maps, num_groups_train, scores_tp=tr_scores_tp)
    candidates_val, _ = cand_gen.generate_candidates(va_tuples, va_scores_rp, valid_set_t.relation_maps, num_groups_val, scores_tp=va_scores_tp)
    candidates_test, scores_test = cand_gen.generate_candidates(te_tuples, te_scores_rp, test_set_t.relation_maps, num_groups_test, scores_tp=te_scores_tp)

    # 5. Build Labels and Hard Negatives
    # Candidates serve as 'Hard Negatives' (high likelihood false positives) for Phase 3.
    # We unite candidates with ground truth to ensure all positives are present.
    candidates_train = torch.unique(torch.cat([candidates_train, train_triples], dim=0), dim=0)
    
    train_labels = triple_lib.build_labels_for_triples(candidates_train, train_triples)
    val_labels = triple_lib.build_labels_for_triples(candidates_val, val_triples)
    test_labels = triple_lib.build_labels_for_triples(candidates_test, test_triples)

    # Log Statistics
    all_triples = torch.cat([train_triples, val_triples, test_triples, candidates_train, candidates_val, candidates_test], dim=0)
    get_group_ids = triple_lib.generate_group_id_function(all_triples, args.group_strategy)
    
    cand_gen.print_candidate_statistics(candidates_train, get_group_ids(candidates_train), train_triples, get_group_ids(train_triples), train_set_t.relation_maps, name="Train")
    cand_gen.print_candidate_statistics(candidates_test, get_group_ids(candidates_test), test_triples, get_group_ids(test_triples), test_set_t.relation_maps, name="Test")

    # 6. Evaluate Candidates (Baseline Metrics - Recall Ceiling)
    stageprint("Evaluating Phase 2 Candidates (Baseline Metrics using Phase 1 scores)...")
    
    # Calculate local true counts for test set metrics
    def get_true_counts_dict(triples):
        g_ids = get_group_ids(triples).to(torch.long)
        counts = torch.bincount(g_ids, minlength=int(g_ids.max().item()) + 1)
        nonzero = (counts > 0).nonzero(as_tuple=False).flatten()
        return {int(i): int(counts[i].item()) for i in nonzero}
    
    test_true_counts = get_true_counts_dict(test_triples)
    
    candidate_results = evaluate_candidate_baseline_metrics(
        scores_test, 
        test_labels, 
        get_group_ids(candidates_test), 
        len(test_triples), 
        test_true_counts
    )
    
    # Save baseline results
    baseline_path = os.path.join(args.log_dir, f"{args.expname}", f"version_{args.version}", "candidate_baseline_results.csv")
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
    pd.DataFrame([candidate_results]).to_csv(baseline_path, index=False)
    logger.info(f"Candidate Baseline Metrics: {candidate_results}")
    
    # 7. Generate Figures (Visual Analysis)
    create_figures(candidates_test, test_triples, test_set_t.relation_maps, train_triples, args.figure_dir)

    return (candidates_train, train_labels), (candidates_val, val_labels), (candidates_test, test_labels), scores_test


def run_phase_3_triple_classification(
    args, 
    train_data, val_data, test_data,
    paths, relcon, tokens_to_idxs,
    train_triples, val_triples, test_triples,
    candidate_scores_test
):
    """
    Executes Phase 3: Triple Classification (Section 5.3.3).
    
    Refines the candidate set S_cand using a structural discriminator.
    Uses weighted BCE to handle class imbalance in candidate sets.
    """
    stageprint("Phase 3: Triple Classification (Refinement on Candidates)")
    
    cand_tr, label_tr = train_data
    cand_va, label_va = val_data
    cand_te, label_te = test_data

    # 1. Configuration Overrides for Phase 3 (Section 5.4.3)
    # Since candidates already contain hard negatives, we disable random negative sampling.
    args_p3 = copy.copy(args)
    if args_p3.num_negatives != 0:
        logger.warning(f"Overriding num_negatives={args_p3.num_negatives} to 0 for Phase 3.")
        args_p3.num_negatives = 0
    if args_p3.val_num_negatives != 0:
        logger.warning(f"Overriding val_num_negatives={args_p3.val_num_negatives} to 0 for Phase 3.")
        args_p3.val_num_negatives = 0
    
    # Configuration overrides for Triple Model stability
    if args_p3.loss_weight != 1.0:
        logger.warning(f"Overriding loss_weight={args_p3.loss_weight} to 1.0 for Phase 3.")
        args_p3.loss_weight = 1.0
    if args_p3.full_test:
        logger.warning(f"Overriding full_test={args_p3.full_test} to False for Phase 3.")
        args_p3.full_test = False
    if args_p3.check_val_every_n_epoch > 1:
        logger.warning(f"Overriding check_val_every_n_epoch={args_p3.check_val_every_n_epoch} to 1 for Phase 3.")
        args_p3.check_val_every_n_epoch = 1

    args_p3.lp_loss_fn = "bce" # Standard Weighted BCE

    # 2. Dataset Setup
    # Uses CandidateTripleEntityMultiPathDataset to handle pre-filtered triples.
    get_group_ids = triple_lib.generate_group_id_function(torch.cat([cand_tr, cand_va, cand_te], dim=0), args.group_strategy)
    
    train_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=paths, relcontext_store=relcon,
        triple_store=cand_tr, labels=label_tr, group_ids=get_group_ids(cand_tr), 
        context_triple_store=train_triples, maximum_triple_paths=args_p3.max_ppt,
        parallel=True, num_workers=args_p3.num_workers, tokens_to_idxs=tokens_to_idxs
    )
    
    # Shared paths
    path_store = (train_set_tri.relation_paths, train_set_tri.entity_paths, train_set_tri.path_index)
    tokens_to_idxs = train_set_tri.tokens_to_idxs

    valid_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        triple_store=cand_va, labels=label_va, group_ids=get_group_ids(cand_va), 
        context_triple_store=train_triples, maximum_triple_paths=args_p3.max_ppt, 
        tokens_to_idxs=tokens_to_idxs, parallel=True, num_workers=args_p3.num_workers
    )
    test_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        triple_store=cand_te, labels=label_te, group_ids=get_group_ids(cand_te), 
        context_triple_store=train_triples, maximum_triple_paths=args_p3.max_ppt, 
        tokens_to_idxs=tokens_to_idxs, parallel=True, num_workers=args_p3.num_workers
    )

    # 3. DataLoaders
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    use_cuda = (args_p3.device == "cuda")
    tr_loader = torch.utils.data.DataLoader(train_set_tri, batch_size=args_p3.batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=use_cuda, num_workers=args_p3.num_workers)
    va_loader = torch.utils.data.DataLoader(valid_set_tri, batch_size=args_p3.val_batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=use_cuda, num_workers=args_p3.num_workers)
    te_loader = torch.utils.data.DataLoader(test_set_tri, batch_size=args_p3.val_batch_size, collate_fn=collate_fn, shuffle=False, pin_memory=use_cuda, num_workers=min(args_p3.num_workers, MAX_WORKERS_TEST))

    # 4. Model Setup
    # PathE encoder trained from scratch for triple classification.
    relcontext_graph = encode_relcontext_freqs(relcon, num_entities=train_triples.max()+1, num_relations=len(tokens_to_idxs)-2, offset=2)
    bundle = partial(bundle_arguments, exclude=["vocab_size"], args=namespace_to_dict(args_p3))
    
    model_tri = PathEModelTriples(
        vocab_size=train_set_tri.vocab_size,
        padding_idx=tokens_to_idxs["PAD"],
        relcontext_graph=relcontext_graph,
        **bundle(target_class=PathEModelTriples),
    )
    
    # Class Balancing weights
    class_weights = triple_lib.get_class_weights_without_special_tokens(train_triples) if args.class_weights else None
    
    pl_model_tri = PathEModelWrapperTriples(
        pathe_model=model_tri,
        filtration_dict=triple_lib.make_relation_filter_dict_no_sp_tokens(train_triples, val_triples, test_triples),
        class_weights=class_weights, 
        **namespace_to_dict(args_p3),
    )
    pl_model_tri.model.to(torch.device(args_p3.device))

    # 5. Metric Injection for Sparse Candidates (Section 5.4.3)
    # We must provide total true positives so recall is calculated against the full KG, not just candidates.
    def count_positives(triples, grouping_fn):
        group_ids = grouping_fn(triples).to(torch.long)
        counts = torch.bincount(group_ids, minlength=int(group_ids.max().item()) + 1)
        nonzero = (counts > 0).nonzero(as_tuple=False).flatten()
        return {int(i): int(counts[i].item()) for i in nonzero}

    val_true_counts = count_positives(val_triples, get_group_ids)
    test_true_counts = count_positives(test_triples, get_group_ids)

    # Inject true counts into the metrics
    for k in pl_model_tri.cand_topk:
        pl_model_tri.cand_metrics_val[f"recall@{k}_perGroup"].set_true_counts(val_true_counts)
        pl_model_tri.cand_metrics_val[f"recall@{k}_total"].set_num_positives(len(val_triples))
        pl_model_tri.cand_metrics_test[f"recall@{k}_perGroup"].set_true_counts(test_true_counts)
        pl_model_tri.cand_metrics_test[f"recall@{k}_total"].set_num_positives(len(test_triples))

    # 6. Trainer Setup & Execution
    tb_logger = TensorBoardLogger(save_dir=args_p3.log_dir, name=args_p3.expname, version=args_p3.version, sub_dir="phase3_triples")
    checkpoint_callback = ModelCheckpoint(
        monitor=args_p3.triple_monitor, dirpath=args_p3.checkpoint_dir, 
        mode=get_metric_mode(pl_model_tri, args_p3.triple_monitor),
        filename=args.model + f"-triple-{{epoch}}-{{{args_p3.triple_monitor}:.2f}}",
        every_n_train_steps=args_p3.chekpoint_ksteps
    )
    
    trainer = Trainer(
        max_epochs=args_p3.max_epochs,
        accelerator="gpu" if args_p3.device == "cuda" else "cpu", devices=args_p3.num_devices, 
        limit_train_batches=(0.1 if args.debug else 1.0),
        logger=[tb_logger],
        callbacks=[EarlyStopping(monitor=args_p3.triple_monitor, patience=args.patience, mode=get_metric_mode(pl_model_tri, args_p3.triple_monitor)), checkpoint_callback, DatasetUpdater([train_set_tri, valid_set_tri] if args_p3.train_paths else [train_set_tri.dataset])],
        gradient_clip_val=1.0
    )

    if args_p3.cmd in ["train", "resume"] and not args_p3.skip_phase2:
        trainer.fit(pl_model_tri, tr_loader, va_loader, ckpt_path=args_p3.triple_checkpoint)
        triple_ckpt = checkpoint_callback.best_model_path
        logger.info(f"[Phase 3] Done. Best model saved in {triple_ckpt}")
    else:
        triple_ckpt = args_p3.triple_checkpoint

    # Final Evaluation
    test_dict = trainer.test(model=pl_model_tri, dataloaders=te_loader, ckpt_path=triple_ckpt)[0]
    pd.DataFrame([test_dict]).to_csv(os.path.join(tb_logger.log_dir, "final_results_summary.csv"), index=False)
    
    return test_dict


def create_and_run_split_join_predict_experiment(args):
    """
    Main entry point for the Split-Join-Predict framework (Chapter 5).
    
    Orchestrates the three phases:
      Phase 1: Property Prediction (Entity-Centric Learning)
      Phase 2: Candidate Generation (Global Join)
      Phase 3: Triple Classification (Refinement)
    """
    start_time = datetime.datetime.now()
    stageprint(f"Starting Split-Join-Predict Experiment ({args.expname}) at: {start_time.strftime('%H:%M:%S')}")

    # --- Global Data Loading ---
    train_tuples, val_tuples, test_tuples = load_tuple_tensors(args.train_paths, args.valid_paths, args.test_paths)
    train_triples, val_triples, test_triples = load_triple_tensors(args.train_paths, args.valid_paths, args.test_paths)
    paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)
    
    # Load and filter inverse relations
    train_rel2inv = du.load_relation2inverse_relation_from_file(args.train_paths)
    val_rel2inv = du.load_relation2inverse_relation_from_file(args.valid_paths)
    test_rel2inv = du.load_relation2inverse_relation_from_file(args.test_paths)
    
    # Ensure relations are consistent
    unique_rels = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
    tokens_to_idxs = create_vocabulary_from_relations(unique_rels.tolist(), ["MSK"])

    # --- Phase 1: Property Prediction ---
    p1_artifacts = run_phase_1_property_prediction(
        args, 
        train_tuples, val_tuples, test_tuples, 
        train_triples, val_triples, test_triples,
        paths, relcon, tokens_to_idxs,
        train_rel2inv, val_rel2inv, test_rel2inv
    )
    trainer_t, pl_model_t, tuple_ckpt, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t = p1_artifacts

    # --- Phase 2: Candidate Generation ---
    cand_train, cand_val, cand_test, cand_scores = run_phase_2_candidate_generation(
        args, trainer_t, pl_model_t, tuple_ckpt,
        tr_loader, va_loader, te_loader,
        train_set_t, valid_set_t, test_set_t,
        train_triples, val_triples, test_triples
    )
    
    # Clean up Phase 1 artifacts to free memory for Phase 3
    del trainer_t, pl_model_t, tr_loader, va_loader, te_loader, train_set_t, valid_set_t, test_set_t
    gc.collect()

    # --- Phase 3: Triple Classification ---
    run_phase_3_triple_classification(
        args, 
        cand_train, cand_val, cand_test,
        paths, relcon, tokens_to_idxs,
        train_triples, val_triples, test_triples,
        cand_scores
    )

    print("Experiment Finished.")