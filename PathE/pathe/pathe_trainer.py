"""
Scalable routines for training a PathE model.

"""
import copy
import os
import logging
import datetime
from functools import partial
from typing import Callable

import torch
torch.set_float32_matmul_precision('high') # Set high precision for matrix multiplications for fast training on tensor cores
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
import gc

from .candidates import *

from . import triple_lib
from .pather_models import PathEModelTriples, PathEModelTuples
from .pathdata import NegativeTripleSampler, TripleEntityMultiPathDataset, TupleEntityMultiPathDataset, CandidateTripleEntityMultiPathDataset, create_vocabulary_from_relations
from .data_utils import collate_multipaths, load_triple_tensors, \
    load_unrolled_setup, load_corrupted_triples_from_dir, \
    memmap_corrupted_triples_from_dir
from .data_utils import load_tuple_tensors
from .callbacks import DatasetUpdater
from .corruption import CorruptHeadGenerator, CorruptHeadGeneratorTuples, CorruptRelationGeneratorTuples, CorruptTailGenerator, CorruptLinkGenerator#, \
    # CorruptBothGenerator
from .utils import stageprint, bundle_arguments, namespace_to_dict
from .wrappers import PathEModelWrapperTriples, PathEModelWrapperTuples
from .path_lib import encode_relcontext_freqs
from . import data_utils as du
from .figures import create_candidate_figures

logger = logging.getLogger(__name__)

# Global max workers to avoid excessive spawning when num_workers is large
MAX_WORKERS_PREDICTION = 16
MAX_WORKERS_TEST = 16


def predict_all(trainer, model, loader, ckpt_path=None):
    # Create a new dataloader without persistent workers for more speed
    pred_loader = torch.utils.data.DataLoader(
        loader.dataset, batch_size=loader.batch_size, 
        collate_fn=loader.collate_fn, shuffle=False,
        pin_memory=loader.pin_memory, 
        num_workers=min(loader.num_workers, MAX_WORKERS_PREDICTION),  # Cap workers globally
        persistent_workers=False # disable persistent workers for prediction as they are not needed afterwards
    )
    outs = trainer.predict(model, dataloaders=pred_loader, ckpt_path=ckpt_path)

    tuples_all = torch.cat([o["tuples"].cpu() for o in outs], dim=0)
    logits_all = torch.cat([o["logits_rp"].cpu() for o in outs], dim=0)
    logits_tp_all = torch.cat([o["logits_tp"].cpu() for o in outs], dim=0)
    
    del outs, pred_loader
    return tuples_all, logits_all, logits_tp_all

def create_and_run_training_exp_tuples(args):
    """
    Main entry point for the experimental setup, execution, and evaluation.
    """
    model_name = args.model
    start_time = datetime.datetime.now()
    stageprint(f"Starting {args.expname} at: {start_time.strftime('%H:%M:%S')}")

    stageprint("Loading data and creating corruptions...")
    # Case A: [UnrolledPathDataset] only for loading
    train_tuples, val_tuples, test_tuples = load_tuple_tensors(
        args.train_paths, args.valid_paths, args.test_paths)
    train_triples, val_triples, test_triples = load_triple_tensors(
        args.train_paths, args.valid_paths, args.test_paths)


    paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)

    # Creating filtration dictionaries and utilities for link prediction
    filtration_dict = triple_lib.make_relation_filter_dict_no_sp_tokens(train_triples, val_triples, test_triples)
    # used in evaluation to filter false negatives
    map_head_to_relationsets_tuples = triple_lib.make_relation_filter_dict_no_sp_tokens_tuples(train_tuples, val_tuples, test_tuples)
    map_relation_to_headsets_tuples = triple_lib.make_head_filter_dict_no_sp_tokens_tuples(train_tuples, val_tuples, test_tuples)
    # Creating the head and tail filtered dict for the triple corruptors
    head_filter_dict, tail_filter_dict = triple_lib.make_head_tail_dicts(train_triples, val_triples, test_triples)
    unique_entities = triple_lib.get_unique_entities(train_triples, val_triples, test_triples)
    num_entities = unique_entities.size()[0]
    unique_relations = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
    tokens_to_idxs = create_vocabulary_from_relations(unique_relations.tolist(), ["MSK"])
    class_weights = triple_lib.get_class_weights_without_special_tokens(train_triples) if args.class_weigths else None

    # Load relation to inverse relation mappings
    train_relation_to_inverse = du.load_relation2inverse_relation_from_file(args.train_paths)
    val_relation_to_inverse = du.load_relation2inverse_relation_from_file(args.valid_paths)
    test_relation_to_inverse = du.load_relation2inverse_relation_from_file(args.test_paths)

    # Get head-tail adjacency matrices for all splits
    train_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(train_triples, num_entities)
    val_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(val_triples, num_entities)
    test_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(test_triples, num_entities)
    # Create global adjacency for filtering
    global_head_tail_adjacency = train_head_tail_adjacency | val_head_tail_adjacency | test_head_tail_adjacency

    # Creating the triple corruptors for head, tail, or both (merged) and
    # preserving the number of positive triples in case of H/T corruption
    tr_positives, va_positives, te_positives = len(train_tuples), len(val_tuples), len(test_tuples)
    tuple_corruptor = None  # instantiated only if num_negatives > 0
    if args.num_negatives:# > 0 and 
        if "r" in args.corruption:
            tuple_corruptor = CorruptRelationGeneratorTuples(map_head_to_relationsets_tuples, entities=unique_entities, relations=unique_relations)
        elif "e" in args.corruption:
            tuple_corruptor = CorruptHeadGeneratorTuples(map_relation_to_headsets_tuples, entities=unique_entities, relations=unique_relations)
        else:
            print(f"Unknown corruption type {args.corruption}, using default relation corruption.")
            tuple_corruptor = CorruptRelationGeneratorTuples(map_head_to_relationsets_tuples, entities=unique_entities, relations=unique_relations)

    parallel = True
    negatives = [None, None, None]  # assume no negative dumpset is available
    if args.dump_dir is not None:
        raise NotImplementedError("Dumping negative tuples is not implemented yet.")
        print(f"Loading checkpointed negative tuples from {args.dump_dir}")
        # The training pos-neg triples are loaded as a numpy memory map 
        negatives[0] = memmap_corrupted_triples_from_dir(
            os.path.join(args.dump_dir, "train"))
        # Validation and test triples are simply loaded as torch tensors
        negatives[1] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "valid"))
        negatives[2] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "test"))

    stageprint("Creating datasets and dataloaders...")
    train_set = TupleEntityMultiPathDataset(
        path_store=paths,
        relcontext_store=relcon,
        tuple_store=train_tuples,
        context_triple_store=train_triples,
        original_relation_to_inverse_relation=train_relation_to_inverse,
        maximum_tuple_paths=args.max_ppt,
        num_negatives=args.num_negatives,
        tuple_corruptor=tuple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_tuple_store=negatives[0],
        head_tail_adjacency=train_head_tail_adjacency,
        tokens_to_idxs=tokens_to_idxs
    )
    # Using shared data structures for valid and test
    tokens_to_idxs = train_set.tokens_to_idxs
    path_store = train_set.relation_paths, \
                 train_set.entity_paths, \
                 train_set.path_index
    valid_set = TupleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        tuple_store=val_tuples,
        context_triple_store=train_triples,
        original_relation_to_inverse_relation=val_relation_to_inverse,
        maximum_tuple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        tuple_corruptor=tuple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_tuple_store=negatives[1],
        head_tail_adjacency=val_head_tail_adjacency
    )
    test_set = TupleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        tuple_store=test_tuples,
        context_triple_store=train_triples,
        original_relation_to_inverse_relation=test_relation_to_inverse,
        maximum_tuple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        tuple_corruptor=tuple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_tuple_store=negatives[2],
        head_tail_adjacency=test_head_tail_adjacency
    )

    if args.num_negatives > 0 and args.dump_negatives:
        raise NotImplementedError("Dumping negative tuples is not implemented yet or at least may not work as expected. Needs to get checked")
        print(f"Dumping negative tuples in {args.dump_dir}")
        for n, d in zip(["tr", "va", "te"], [train_set, valid_set, test_set]):
            negt = d.tuplestore  # FIXME via the dump_negatives()
            torch.save(negt, os.path.join(args.dump_dir, f'{n}_tuple_negatives.pt'))

    print("Found {} samples in the dataset: Tr {}, Va {}, Te {}"
          .format(len(train_set) + len(valid_set) + len(test_set),
                  len(train_set), len(valid_set), len(test_set)))
    print(f"Vocabulary size (with special tokens): {len(tokens_to_idxs)}")

    # Adjust batch sizes to be compatible with number of negatives
    if args.val_batch_size % (args.val_num_negatives + 1) != 0:
        positive_batches = args.val_batch_size // (args.val_num_negatives + 1)
        args.val_batch_size = positive_batches * (args.val_num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping val/test batch size to: {args.val_batch_size}")
    if args.batch_size % (args.num_negatives + 1) != 0:
        positive_batches = args.batch_size // (args.num_negatives + 1)
        args.batch_size = positive_batches * (args.num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping train batch size to: {args.batch_size}")

    # Creating the data loaders for each partition
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    use_cuda = (args.device == "cuda")
    use_persist = args.num_workers > 0
    tr_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist,
        sampler=NegativeTripleSampler(tr_positives, args.num_negatives))
    va_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist,
        sampler=NegativeTripleSampler(va_positives, args.val_num_negatives))
    te_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=use_cuda, num_workers=min(args.num_workers, MAX_WORKERS_TEST),
        persistent_workers=False,
        sampler=NegativeTripleSampler(te_positives, args.val_num_negatives))

    stageprint("Creating model and loading checkpoints")
    # This should assume model and basic parameters
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon,
        num_entities=num_entities,
        num_relations=train_set.vocab_size - 2,
        offset=2,  # applied to both ents and rels
    )
    bundle = partial(bundle_arguments, exclude=["vocab_size"],
                     args=namespace_to_dict(args))
    model = PathEModelTuples(
        vocab_size=train_set.vocab_size,
        relcontext_graph=relcontext_graph,
        padding_idx=tokens_to_idxs["PAD"],
        **bundle(target_class=PathEModelTuples),
    )
    # class_weights = triple_lib.get_class_weights(train_triples, tokens_to_idxs)

    pl_model = PathEModelWrapperTuples(
        pathe_model=model,
        filtration_dict=map_head_to_relationsets_tuples,
        global_head_tail_adjacency=global_head_tail_adjacency,
        train_head_tail_adjacency=train_head_tail_adjacency,
        val_head_tail_adjacency=val_head_tail_adjacency,
        test_head_tail_adjacency=test_head_tail_adjacency,
        class_weights=class_weights,  # for rel imbalance
        **namespace_to_dict(args),  # model hparameters
    )
    # print(pl_model.model)  # keras-style model overview
    pl_model.model.to(torch.device(args.device))

    stageprint("Creating loggers, callbacks and setting up trainer")
    # Registering loggers based on the experiment name and version
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.expname,
        version=args.version)
    exp_dir = tb_logger.log_dir  # equivalent to args.exp_dir
    if args.wandb_project is not None:  # use synchronised wandb logger
        wb_logger = WandbLogger(
            id=args.wandb_id,
            save_dir=exp_dir,
            name=args.expname,
            project=args.wandb_project,
            log_model="all",
            sync_tensorboard=True)
        # Logging gradients, topology, histogram
        wb_logger.watch(pl_model, log="all")
        wb_logger.log_hyperparams(args)
    else: 
        wb_logger = None
        tb_logger.log_hyperparams(args)

    # Creating callbacks for checkpointing and early stopping
    mode = "min" if args.tuple_monitor.endswith("loss") else "max"
    checkpoint_callbk = ModelCheckpoint(
        monitor=args.tuple_monitor, dirpath=args.checkpoint_dir, mode=mode,
        filename=model_name + f"-tuple-{{epoch}}-{{{args.tuple_monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps)
    estopping_callbk = EarlyStopping(
        monitor=args.tuple_monitor, patience=args.patience, mode=mode, check_on_train_epoch_end=False)
    dataset_callbk = DatasetUpdater(  # datasets to seed per-epoch
        [train_set, valid_set] if args.train_paths else [train_set.dataset])

    # accumulated_batches = 5 # {2: 3, 4: 6}
    tr_limit, va_limit = (.1, .2) if args.debug else (1., 1.)
    accelerator = "gpu" if args.device == "cuda" else "cpu"

    logger_tmp = [tb_logger] + ([wb_logger] if wb_logger else [])
    # Automatic gradient clipping and Automatic gradient accumulation is not supported for manual optimization.
    if args.use_manual_optimization:
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator, devices=args.num_devices, num_nodes=1,
            limit_train_batches=tr_limit, limit_val_batches=va_limit,
            logger=logger_tmp, log_every_n_steps=5,
            val_check_interval=args.val_check_interval,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[estopping_callbk, checkpoint_callbk, dataset_callbk],
            num_sanity_val_steps=-1,
        )
    else:
        trainer = Trainer(
            max_epochs=args.max_epochs, gradient_clip_val=1.0,
            accelerator=accelerator, devices=args.num_devices, num_nodes=1,
            limit_train_batches=tr_limit, limit_val_batches=va_limit,
            logger=logger_tmp, log_every_n_steps=5,
            val_check_interval=args.val_check_interval,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            gradient_clip_algorithm='norm',  # CHANGED THIS
            accumulate_grad_batches=args.accumulate_gradient,
            callbacks=[estopping_callbk, checkpoint_callbk, dataset_callbk],
            num_sanity_val_steps=-1,
        )

    if args.cmd in ["train", "resume"]:
        # Train and resume are the same assuming their setup is consistent
        stageprint("Training-validating the model, be patient!")
        trainer.fit(pl_model, tr_dataloader, va_dataloader,
                    ckpt_path=args.tuple_checkpoint)  # load or None
        args.tuple_checkpoint = checkpoint_callbk.best_model_path
        print(f"Done. Best model saved in {args.tuple_checkpoint}")
        ttime = (datetime.datetime.now() - start_time).total_seconds() / 3600
        print(f"Training time: {round(ttime, 2)} hours")

    # stageprint("Evaluating the model on the validation set")
    # valid_dict = trainer.validate(
    #     dataloaders=va_dataloader,
    #     ckpt_path=checkpoint_callbk.best_model_path)[0]
    # del va_dataloader, valid_set  # free some memory
    # print("\nValidation results: {}".format(valid_dict))

    stageprint("Evaluating the model on the test set")
    test_dict = trainer.test(model=pl_model if args.cmd == "test" else None,
                             dataloaders=te_dataloader,
                             ckpt_path=args.tuple_checkpoint)[0]
    print("\nTesting results: {}".format(test_dict))

    # results_dict = {**valid_dict, **test_dict}
    # results_dict.update(namespace_to_dict(args))  # +hparams
    # fname = os.path.join(args.log_dir, "results_summary.csv")
    # write_csv(results_dict, fname)
    # results_raw = pd.read_csv(fname)

    # return results_dict, results_raw

def create_and_run_training_exp_triples(args):
    """
    Main entry point for the experimental setup, execution, and evaluation.
    """
    model_name = args.model
    start_time = datetime.datetime.now()
    stageprint(f"Starting {args.expname} at: {start_time.strftime('%H:%M:%S')}")

    stageprint("Loading data and creating corruptions...")
    # Case A: [UnrolledPathDataset] only for loading
    train_triples, val_triples, test_triples = load_triple_tensors(
        args.train_paths, args.valid_paths, args.test_paths)
    paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)

    # Creating filtration dictionaries and utilities for link prediction
    filtration_dict = triple_lib.make_relation_filter_dict_no_sp_tokens(
        train_triples, val_triples, test_triples)
    # Creating the head and tail filtered dict for the triple corruptors
    head_filter_dict, tail_filter_dict = triple_lib.make_head_tail_dicts(
        train_triples, val_triples, test_triples)
    unique_entities = triple_lib.get_unique_entities(
        train_triples, val_triples, test_triples)
    num_entities = unique_entities.size()[0]
    unique_relations = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
    tokens_to_idxs = create_vocabulary_from_relations(unique_relations.tolist(), ["MSK"])
    class_weights = triple_lib.get_class_weights_without_special_tokens(
        train_triples) if args.class_weigths else None

    # Creating the triple corruptors for head, tail, or both (merged) and
    # preserving the number of positive triples in case of H/T corruption
    tr_positives, va_positives, te_positives = \
        len(train_triples), len(val_triples), len(test_triples)
    triple_corruptor = None  # instantiated only if num_negatives > 0
    if args.num_negatives > 0 and args.corruption in ["h", "t"]:
        triple_corruptor = CorruptHeadGenerator(
            filter_dict=head_filter_dict, entities=unique_entities) \
            if args.corruption == "h" else CorruptTailGenerator(
            filter_dict=tail_filter_dict, entities=unique_entities)
    elif args.num_negatives > 0:  # "all" applies and merge both corruptions
        triple_corruptor = CorruptLinkGenerator(
            head_filter_dict=head_filter_dict,
            tail_filter_dict=tail_filter_dict,
            entities=unique_entities)
        # The number of positive is simply duplicated to measure LP results
        tr_positives, va_positives, te_positives = \
            [i * 2 for i in [tr_positives, va_positives, te_positives]]

    parallel = True
    negatives = [None, None, None]  # assume no negative dumpset is available
    if args.dump_dir is not None:
        print(f"Loading checkpointed negative triples from {args.dump_dir}")
        # The training pos-neg triples are loaded as a numpy memory map 
        negatives[0] = memmap_corrupted_triples_from_dir(
            os.path.join(args.dump_dir, "train"))
        # Validation and test triples are simply loaded as torch tensors
        negatives[1] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "valid"))
        negatives[2] = load_corrupted_triples_from_dir(
                        os.path.join(args.dump_dir, "test"))

    stageprint("Creating datasets and dataloaders...")
    train_set = TripleEntityMultiPathDataset(
        path_store=paths,
        relcontext_store=relcon,
        triple_store=train_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        num_negatives=args.num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_triple_store=negatives[0],
        tokens_to_idxs=tokens_to_idxs
    )
    # Using shared data structures for valid and test
    tokens_to_idxs = train_set.tokens_to_idxs
    path_store = train_set.relation_paths, \
                 train_set.entity_paths, \
                 train_set.path_index
    valid_set = TripleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        triple_store=val_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_triple_store=negatives[1],
    )
    test_set = TripleEntityMultiPathDataset(
        path_store=path_store,
        relcontext_store=relcon,
        triple_store=test_triples,
        context_triple_store=train_triples,
        maximum_triple_paths=args.max_ppt,
        tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives,
        triple_corruptor=triple_corruptor,
        parallel=parallel,
        num_workers=args.num_workers,
        neg_triple_store=negatives[2],
    )

    if args.num_negatives > 0 and args.dump_negatives:
        print(f"Dumping negative triples in {args.dump_dir}")
        for n, d in zip(["tr", "va", "te"], [train_set, valid_set, test_set]):
            negt = d.triplestore  # FIXME via the dump_negatives()
            torch.save(negt, os.path.join(args.dump_dir, f'{n}_negatives.pt'))

    print("Found {} samples in the dataset: Tr {}, Va {}, Te {}"
          .format(len(train_set) + len(valid_set) + len(test_set),
                  len(train_set), len(valid_set), len(test_set)))
    print(f"Vocabulary size (with special tokens): {len(tokens_to_idxs)}")

    if args.val_batch_size % (args.val_num_negatives + 1) != 0:
        positive_batches = args.val_batch_size // (args.val_num_negatives + 1)
        args.val_batch_size = positive_batches * (args.val_num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping val/test batch size to:"
                       f" {args.val_batch_size}")
    if args.batch_size % (args.num_negatives + 1) != 0:
        positive_batches = args.batch_size // (args.num_negatives + 1)
        args.batch_size = positive_batches * (args.num_negatives + 1)
        # discarded_size = args.batch_size - fixed_bsize
        logger.warning(f"Clamping train batch size to: {args.batch_size}")

    # Creating the data loaders for each partition
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    use_cuda = (args.device == "cuda")
    use_persist = args.num_workers > 0
    tr_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=True,
        pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist,
        sampler=NegativeTripleSampler(tr_positives, args.num_negatives))
    va_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist,
        sampler=NegativeTripleSampler(va_positives, args.val_num_negatives))
    te_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size,
        collate_fn=collate_fn, shuffle=False,
        pin_memory=use_cuda, num_workers=min(args.num_workers, MAX_WORKERS_TEST),
        persistent_workers=False,
        sampler=NegativeTripleSampler(te_positives, args.val_num_negatives))

    stageprint("Creating model and loading checkpoints")
    # This should assume model and basic parameters
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon,
        num_entities=num_entities,
        num_relations=train_set.vocab_size - 2,
        offset=2,  # applied to both ents and rels
    )
    bundle = partial(bundle_arguments, exclude=["vocab_size"],
                     args=namespace_to_dict(args))
    model = PathEModelTriples(
        vocab_size=train_set.vocab_size,
        padding_idx=tokens_to_idxs["PAD"],
        relcontext_graph=relcontext_graph,
        **bundle(target_class=PathEModelTriples),
    )
    # class_weights = triple_lib.get_class_weights(train_triples, tokens_to_idxs)
    pl_model = PathEModelWrapperTriples(
        pathe_model=model,
        filtration_dict=filtration_dict,  # FIXME
        class_weights=class_weights,  # for rel imbalance
        **namespace_to_dict(args),  # model hparameters
    )
    # print(pl_model.model)  # keras-style model overview
    pl_model.model.to(torch.device(args.device))

    stageprint("Creating loggers, callbacks and setting up trainer")
    # Registering loggers based on the experiment name and version
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.expname,
        version=args.version)
    exp_dir = tb_logger.log_dir  # equivalent to args.exp_dir
    if args.wandb_project is not None:  # use synchronised wandb logger
        wb_logger = WandbLogger(
            id=args.wandb_id,
            save_dir=exp_dir,
            name=args.expname,
            project=args.wandb_project,
            log_model="all",
            sync_tensorboard=True)
        # Logging gradients, topology, histogram
        wb_logger.watch(pl_model, log="all")
        wb_logger.log_hyperparams(args)
    else: 
        wb_logger = None
        tb_logger.log_hyperparams(args)

    # Creating callbacks for checkpointing and early stopping
    mode = "min" if args.triple_monitor.endswith("loss") else "max"
    checkpoint_callbk = ModelCheckpoint(
        monitor=args.triple_monitor, dirpath=args.checkpoint_dir, mode=mode,
        filename=model_name + f"-triple-{{epoch}}-{{{args.triple_monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps)
    estopping_callbk = EarlyStopping(
        monitor=args.triple_monitor, patience=args.patience, mode=mode, check_on_train_epoch_end=False)
    dataset_callbk = DatasetUpdater(  # datasets to seed per-epoch
        [train_set, valid_set] if args.train_paths else [train_set.dataset])

    # accumulated_batches = 5 # {2: 3, 4: 6}
    tr_limit, va_limit = (.1, .2) if args.debug else (1., 1.)
    accelerator = "gpu" if args.device == "cuda" else "cpu"

    logger_tmp = [tb_logger] + ([wb_logger] if wb_logger else [])
    trainer = Trainer(
        max_epochs=args.max_epochs, gradient_clip_val=1.0,
        accelerator=accelerator, devices=args.num_devices, num_nodes=1,
        limit_train_batches=tr_limit, limit_val_batches=va_limit,
        logger=logger_tmp, log_every_n_steps=5,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_algorithm='norm',  # CHANGED THIS
        accumulate_grad_batches=args.accumulate_gradient,
        callbacks=[estopping_callbk, checkpoint_callbk, dataset_callbk],
        num_sanity_val_steps=-1,
    )

    if args.cmd in ["train", "resume"]:
        # Train and resume are the same assuming their setup is consistent
        stageprint("Training-validating the model, be patient!")
        trainer.fit(pl_model, tr_dataloader, va_dataloader,
                    ckpt_path=args.triple_checkpoint)  # load or None
        args.triple_checkpoint = checkpoint_callbk.best_model_path
        print(f"Done. Best model saved in {args.triple_checkpoint}")
        ttime = (datetime.datetime.now() - start_time).total_seconds() / 3600
        print(f"Training time: {round(ttime, 2)} hours")

    # stageprint("Evaluating the model on the validation set")
    # valid_dict = trainer.validate(
    #     dataloaders=va_dataloader,
    #     ckpt_path=checkpoint_callbk.best_model_path)[0]
    # del va_dataloader, valid_set  # free some memory
    # print("\nValidation results: {}".format(valid_dict))

    stageprint("Evaluating the model on the test set")
    test_dict = trainer.test(model=pl_model if args.cmd == "test" else None,
                             dataloaders=te_dataloader,
                             ckpt_path=args.triple_checkpoint)[0]
    print("\nTesting results: {}".format(test_dict))

    # results_dict = {**valid_dict, **test_dict}
    # results_dict.update(namespace_to_dict(args))  # +hparams
    # fname = os.path.join(args.log_dir, "results_summary.csv")
    # write_csv(results_dict, fname)
    # results_raw = pd.read_csv(fname)

    # return results_dict, results_raw

def create_and_run_training_exp_two_phases(args):
    """
    Main entry point for the experimental setup, execution, and evaluation.
    Two-phase experiment without negatives.
      Phase 1a: Train the tuple model (num_negatives=0).
      Phase 1b: Predict all tuple logits; build triple candidates globally.
      Phase 3 : Train a triple model on candidate triples (num_negatives=0), evaluate on original test triples.
    """
    model_name = args.model
    start_phase1_time = datetime.datetime.now()
    stageprint(f"Starting {args.expname} (two phases) at: {start_phase1_time.strftime('%H:%M:%S')}")

    # ---------------------------
    # Load data
    # ---------------------------
    stageprint("Loading data...")
    train_tuples, val_tuples, test_tuples = load_tuple_tensors(args.train_paths, args.valid_paths, args.test_paths)
    train_triples, val_triples, test_triples = load_triple_tensors(args.train_paths, args.valid_paths, args.test_paths)


    paths, relcon, _ = du.load_unrolled_setup(args.train_paths, args.path_setup)
    filtration_dict = triple_lib.make_relation_filter_dict_no_sp_tokens(train_triples, val_triples, test_triples)
    unique_entities = triple_lib.get_unique_entities(train_triples, val_triples, test_triples)
    assert unique_entities.size(0) == unique_entities.max() + 1, "Entity IDs must be contiguous and start from 0"
    num_entities = unique_entities.size(0)
    unique_relations = triple_lib.get_unique_relations(train_triples, val_triples, test_triples)
    tokens_to_idxs = create_vocabulary_from_relations(unique_relations.tolist(), ["MSK"])
    class_weights = triple_lib.get_class_weights_without_special_tokens(train_triples) if args.class_weigths else None

    # Load relation to inverse relation mappings
    train_rel2inv = du.load_relation2inverse_relation_from_file(args.train_paths)
    val_rel2inv   = du.load_relation2inverse_relation_from_file(args.valid_paths)
    test_rel2inv  = du.load_relation2inverse_relation_from_file(args.test_paths)

    # Filter out inverse relations from triples
    train_triples = train_triples[torch.isin(train_triples[:, 1], torch.tensor(list(train_rel2inv.keys()), dtype=torch.long, device=train_triples.device))]
    val_triples   = val_triples[torch.isin(val_triples[:, 1], torch.tensor(list(val_rel2inv.keys()), dtype=torch.long, device=val_triples.device))]
    test_triples  = test_triples[torch.isin(test_triples[:, 1], torch.tensor(list(test_rel2inv.keys()), dtype=torch.long, device=test_triples.device))]

    # Get head-tail adjacency matrices for all splits
    train_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(train_triples, num_entities)
    val_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(val_triples, num_entities)
    test_head_tail_adjacency = triple_lib.get_full_adjacency_matrix(test_triples, num_entities)
    # Create global adjacency for filtering
    global_head_tail_adjacency = train_head_tail_adjacency | val_head_tail_adjacency | test_head_tail_adjacency

    assert(args.num_negatives == 0), "This two-phase training only works with num_negatives=0"
    assert(args.val_num_negatives == 0), "This two-phase training only works with val_num_negatives=0"

    # ---------------------------
    # Phase 1a: Tuple training (no negatives)
    # ---------------------------
    stageprint("Phase 1a: Creating datasets and dataloaders...")
    parallel = True

    train_set_t = TupleEntityMultiPathDataset(
        path_store=paths, relcontext_store=relcon,
        tuple_store=train_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=train_rel2inv,
        maximum_tuple_paths=args.max_ppt,
        num_negatives=args.num_negatives, tuple_corruptor=None,
        parallel=parallel, num_workers=args.num_workers, neg_tuple_store=None,
        head_tail_adjacency=train_head_tail_adjacency,
        tokens_to_idxs=tokens_to_idxs
    )
    # Using shared data structures for valid and test
    tokens_to_idxs = train_set_t.tokens_to_idxs
    path_store = (train_set_t.relation_paths, train_set_t.entity_paths, train_set_t.path_index)

    valid_set_t = TupleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        tuple_store=val_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=val_rel2inv,
        maximum_tuple_paths=args.max_ppt, tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives, tuple_corruptor=None,
        parallel=parallel, num_workers=args.num_workers, neg_tuple_store=None,
        head_tail_adjacency=val_head_tail_adjacency
    )
    test_set_t = TupleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        tuple_store=test_tuples, context_triple_store=train_triples,
        original_relation_to_inverse_relation=test_rel2inv,
        maximum_tuple_paths=args.max_ppt, tokens_to_idxs=tokens_to_idxs,
        num_negatives=args.val_num_negatives, tuple_corruptor=None,
        parallel=parallel, num_workers=args.num_workers, neg_tuple_store=None,
        head_tail_adjacency=test_head_tail_adjacency
    )

    print(f"Found {len(train_set_t) + len(valid_set_t) + len(test_set_t)} samples in the dataset: Tr {len(train_set_t)}, Va {len(valid_set_t)}, Te {len(test_set_t)}")
    print(f"Vocabulary size (with special tokens): {len(tokens_to_idxs)}")

    # Creating the data loaders for each partition
    collate_fn = partial(collate_multipaths, padding_idx=tokens_to_idxs["PAD"])
    use_cuda = (args.device == "cuda")
    use_persist = args.num_workers > 0
    tr_loader_t = torch.utils.data.DataLoader(
        train_set_t, batch_size=args.batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist)
    va_loader_t = torch.utils.data.DataLoader(
        valid_set_t, batch_size=args.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=args.num_workers,
        persistent_workers=use_persist)
    te_loader_t = torch.utils.data.DataLoader(
        test_set_t, batch_size=args.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda, num_workers=min(args.num_workers, MAX_WORKERS_TEST),
        persistent_workers=False)

    # Model + wrapper
    stageprint("Creating model and loading checkpoints")
    relcontext_graph = encode_relcontext_freqs(
        relcontext=relcon, num_entities=num_entities,
        num_relations=train_set_t.vocab_size - 2, offset=2)
    bundle = partial(bundle_arguments, exclude=["vocab_size"], args=namespace_to_dict(args))
    model_t = PathEModelTuples(
        vocab_size=train_set_t.vocab_size,
        relcontext_graph=relcontext_graph,
        padding_idx=tokens_to_idxs["PAD"],
        **bundle(target_class=PathEModelTuples),
    )
    # tuple-specific filters for metrics
    map_head_to_relsets = triple_lib.make_relation_filter_dict_no_sp_tokens_tuples(train_tuples, val_tuples, test_tuples)

    pl_model_t = PathEModelWrapperTuples(
        pathe_model=model_t,
        filtration_dict=map_head_to_relsets,
        global_head_tail_adjacency=global_head_tail_adjacency,
        train_head_tail_adjacency=train_head_tail_adjacency,
        val_head_tail_adjacency=val_head_tail_adjacency,
        test_head_tail_adjacency=test_head_tail_adjacency,
        class_weights=class_weights,
        **namespace_to_dict(args),
    )
    pl_model_t.model.to(torch.device(args.device))

    stageprint("Creating loggers, callbacks and setting up trainer")
    # Loggers and trainer
    tb_logger_t = TensorBoardLogger(save_dir=args.log_dir, name=args.expname, version=args.version, sub_dir="tuples")
    exp_dir_t = tb_logger_t.log_dir
    if args.wandb_project is not None:
        wb_logger_t = WandbLogger(id=args.wandb_id, save_dir=exp_dir_t, name=f"{args.expname}_tuples",
                                  project=args.wandb_project, log_model="all", sync_tensorboard=True)
        wb_logger_t.watch(pl_model_t, log="all"); wb_logger_t.log_hyperparams(args)
    else:
        wb_logger_t = None
        tb_logger_t.log_hyperparams(args)

    # Creating callbacks for checkpointing and early stopping
    mode = "min" if args.tuple_monitor.endswith("loss") else "max"
    checkpoint_callbk_t = ModelCheckpoint(
        monitor=args.tuple_monitor, dirpath=args.checkpoint_dir, mode=mode,
        filename=model_name + f"-tuple-{{epoch}}-{{{args.tuple_monitor}:.2f}}",
        every_n_train_steps=args.chekpoint_ksteps)
    estopping_callbk_t = EarlyStopping(
        monitor=args.tuple_monitor, patience=args.patience, mode=mode, check_on_train_epoch_end=False)
    dataset_callbk_t = DatasetUpdater([train_set_t, valid_set_t] if args.train_paths else [train_set_t.dataset])

    tr_limit, va_limit = (.1, .2) if args.debug else (1., 1.)
    accelerator = "gpu" if args.device == "cuda" else "cpu"
    # Automatic gradient clipping and Automatic gradient accumulation is not supported for manual optimization.
    if args.use_manual_optimization:
        trainer_t = Trainer(
            max_epochs=args.max_epochs,
            accelerator=accelerator, devices=args.num_devices, num_nodes=1,
            limit_train_batches=tr_limit, limit_val_batches=va_limit,
            logger=[tb_logger_t] + ([wb_logger_t] if wb_logger_t else []), log_every_n_steps=5,
            val_check_interval=args.val_check_interval,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=[estopping_callbk_t, checkpoint_callbk_t, dataset_callbk_t],
            num_sanity_val_steps=-1,
        )
    else:
        trainer_t = Trainer(
            max_epochs=args.max_epochs, gradient_clip_val=1.0,
            accelerator=accelerator, devices=args.num_devices, num_nodes=1,
            limit_train_batches=tr_limit, limit_val_batches=va_limit,
            logger=[tb_logger_t] + ([wb_logger_t] if wb_logger_t else []), log_every_n_steps=5,
            val_check_interval=args.val_check_interval,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            gradient_clip_algorithm='norm',  # CHANGED THIS
            accumulate_grad_batches=args.accumulate_gradient,
            callbacks=[estopping_callbk_t, checkpoint_callbk_t, dataset_callbk_t],
            num_sanity_val_steps=-1,
        )
    if args.cmd in ["train", "resume"] and not args.skip_phase1:
        # Train and resume are the same assuming their setup is consistent
        stageprint("Training-validating the model, be patient!")
        trainer_t.fit(pl_model_t, tr_loader_t, va_loader_t, ckpt_path=args.tuple_checkpoint)
        tuple_ckpt = checkpoint_callbk_t.best_model_path
        print(f"[Tuples] Done. Best model saved in {tuple_ckpt}")
        ttime = (datetime.datetime.now() - start_phase1_time).total_seconds() / 3600
        print(f"[Tuples] Training time: {round(ttime, 2)} hours")
    else:
        # For inference-only runs
        tuple_ckpt = args.tuple_checkpoint
        print(f"[Tuples] Using checkpoint for prediction: {tuple_ckpt}")
        if args.cmd == "test" and not tuple_ckpt:
            raise ValueError("tuple_checkpoint is required for cmd='test'.")
            
    if not args.skip_phase1:
        stageprint("Evaluating the tuple model (relation predictor) on the test set")
        tuple_test_dict = trainer_t.test(
            model=pl_model_t if args.cmd == "test" else None,
            dataloaders=te_loader_t,
            ckpt_path=tuple_ckpt
        )[0]
        print("\nTesting results (tuple model): {}".format(tuple_test_dict))

    # ---------------------------
    # Phase 1b: Global candidate generation (predict over ALL tuples)
    # ---------------------------
    stageprint(f"Phase 1b: Predicting over all tuples and building candidates...")

    tr_tuples_all, tr_logits_all, tr_logits_tp_all = predict_all(trainer_t, pl_model_t, tr_loader_t, ckpt_path=tuple_ckpt)
    va_tuples_all, va_logits_all, va_logits_tp_all = predict_all(trainer_t, pl_model_t, va_loader_t, ckpt_path=tuple_ckpt)
    te_tuples_all, te_logits_all, te_logits_tp_all = predict_all(trainer_t, pl_model_t, te_loader_t, ckpt_path=tuple_ckpt)

    # Instantiate candidate generator based on args.candidate_generator
    if args.candidate_generator == 'global':
        candidate_generator = CandidateGeneratorGlobal(p=args.candidates_threshold_p, q=args.candidates_quantile_q, temperature=args.candidates_temperature, alpha=args.candidates_alpha, per_group_cap=args.candidates_cap, normalize_mode=args.candidates_normalize_mode, max_num_workers=args.num_workers)
    elif args.candidate_generator == 'global_with_tail':
        candidate_generator = CandidateGeneratorGlobalWithTail(p=args.candidates_threshold_p, q=args.candidates_quantile_q, temperature=args.candidates_temperature, alpha=args.candidates_alpha, beta=args.candidates_beta, per_group_cap=args.candidates_cap, normalize_mode=args.candidates_normalize_mode, max_num_workers=args.num_workers)
    elif args.candidate_generator == 'per_head':
        candidate_generator = CandidateGeneratorPerHead(per_group_cap=args.candidates_cap, alpha=args.candidates_alpha)
    else:
        raise ValueError(f"Unknown candidate_generator: {args.candidate_generator}")

    # Run grid search on test set to find best alpha, beta, temperature
    best_params_total, best_params_per_group = grid_search_candidates(
        candidate_generator,
        args, 
        tr_tuples_all, tr_logits_all, tr_logits_tp_all, 
        va_tuples_all, va_logits_all, va_logits_tp_all, 
        te_tuples_all, te_logits_all, te_logits_tp_all, 
        train_triples, val_triples, test_triples, 
        train_set_t, valid_set_t, test_set_t
    )
    # Run grid search over candidate sizes
    grid_search_candidate_sizes(
        candidate_generator,
        args, 
        tr_tuples_all, tr_logits_all, tr_logits_tp_all, 
        va_tuples_all, va_logits_all, va_logits_tp_all, 
        te_tuples_all, te_logits_all, te_logits_tp_all, 
        train_triples, val_triples, test_triples, 
        train_set_t, valid_set_t, test_set_t
    )

    # Compute number of groups for each split based on triples
    num_groups_train = len(torch.unique(train_triples[:, args.group_strategy], dim=0))
    num_groups_val = len(torch.unique(val_triples[:, args.group_strategy], dim=0))
    num_groups_test = len(torch.unique(test_triples[:, args.group_strategy], dim=0))

    candidates_train, _scores_train = candidate_generator.generate_candidates(tr_tuples_all, tr_logits_all, train_set_t.relation_maps, num_groups_train, logits_tp=tr_logits_tp_all)
    candidates_val, _scores_val = candidate_generator.generate_candidates(va_tuples_all, va_logits_all, valid_set_t.relation_maps, num_groups_val, logits_tp=va_logits_tp_all)
    candidates_test, _scores_test = candidate_generator.generate_candidates(te_tuples_all, te_logits_all, test_set_t.relation_maps, num_groups_test, logits_tp=te_logits_tp_all)

    del tr_tuples_all, tr_logits_all, tr_logits_tp_all
    del va_tuples_all, va_logits_all, va_logits_tp_all
    del te_tuples_all, te_logits_all, te_logits_tp_all

    # add true triples to train candidates (if not already present) to ensure all positives are included
    candidates_train = torch.unique(torch.cat([candidates_train, train_triples], dim=0), dim=0)

    # build labels for candidates
    train_labels = triple_lib.build_labels_for_triples(candidates_train, train_triples)
    val_labels = triple_lib.build_labels_for_triples(candidates_val, val_triples)
    test_labels = triple_lib.build_labels_for_triples(candidates_test, test_triples)

    # create group mappings for group ids
    all_triples = torch.cat([train_triples, val_triples, test_triples, candidates_train, candidates_val, candidates_test], dim=0)
    get_group_ids = triple_lib.generate_group_id_function(all_triples, args.group_strategy)
    del all_triples # free memory

    # Candidate statistics
    candidate_generator.print_candidate_statistics(candidates_train, get_group_ids(candidates_train), train_triples, get_group_ids(train_triples), train_set_t.relation_maps, name="Train")
    print()
    candidate_generator.print_candidate_statistics(candidates_val, get_group_ids(candidates_val), val_triples, get_group_ids(val_triples), valid_set_t.relation_maps, name="Val")
    print()
    candidate_generator.print_candidate_statistics(candidates_test, get_group_ids(candidates_test), test_triples, get_group_ids(test_triples), test_set_t.relation_maps, name="Test")

    # Create candidate figures
    create_candidate_figures(candidates_test, test_triples, test_set_t.relation_maps, train_triples, args.figure_dir)

    # Cleanup Phase 1 resources to stop workers and free memory
    del tr_loader_t, va_loader_t, te_loader_t
    del train_set_t, valid_set_t, test_set_t
    del candidate_generator
    if args.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # ---------------------------
    # Phase 3: Triple training on candidates (no negatives) and test on gold
    # ---------------------------
    start_phase3_time = datetime.datetime.now()
    stageprint("Phase 3: Training triple model on candidate triples (no negatives)...")
    # Adjust args for triple training
    args_phase3 = copy.copy(args)
    if args_phase3.num_negatives != 0:
        logger.warning(f"Overriding num_negatives={args_phase3.num_negatives} to 0 for candidate training in phase 3.")
        args_phase3.num_negatives = 0
    if args_phase3.val_num_negatives != 0:
        logger.warning(f"Overriding val_num_negatives={args_phase3.val_num_negatives} to 0 for candidate training in phase 3.")
        args_phase3.val_num_negatives = 0
    if args_phase3.loss_weight != 1.0:
        logger.warning(f"Overriding loss_weight={args_phase3.loss_weight} to 1.0 for candidate training in phase 3.")
        args_phase3.loss_weight = 1.0
    if args_phase3.full_test:
        logger.warning(f"Overriding full_test={args_phase3.full_test} to False for candidate training in phase 3.")
        args_phase3.full_test = False
    if args_phase3.lp_loss_fn != "bce":
        logger.warning(f"Overriding lp_loss_fn={args_phase3.lp_loss_fn} to 'bce' for candidate training in phase 3.")
        args_phase3.lp_loss_fn = "bce"  # BCE for positives + negatives in candidates
    
    # path_store = (train_set_t.relation_paths, train_set_t.entity_paths, train_set_t.path_index)
    train_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=paths, relcontext_store=relcon,
        triple_store=candidates_train, labels=train_labels, group_ids=get_group_ids(candidates_train), 
        context_triple_store=train_triples, maximum_triple_paths=args_phase3.max_ppt,
        parallel=parallel, num_workers=args_phase3.num_workers, tokens_to_idxs=tokens_to_idxs)
    tokens_to_idxs = train_set_tri.tokens_to_idxs  # shared
    path_store = (train_set_tri.relation_paths, train_set_tri.entity_paths, train_set_tri.path_index)
    valid_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        triple_store=candidates_val, labels=val_labels, group_ids=get_group_ids(candidates_val), 
        context_triple_store=train_triples, maximum_triple_paths=args_phase3.max_ppt, 
        tokens_to_idxs=tokens_to_idxs, parallel=parallel, num_workers=args_phase3.num_workers)
    test_set_tri = CandidateTripleEntityMultiPathDataset(
        path_store=path_store, relcontext_store=relcon,
        triple_store=candidates_test, labels=test_labels, group_ids=get_group_ids(candidates_test), 
        context_triple_store=train_triples, maximum_triple_paths=args_phase3.max_ppt, 
        tokens_to_idxs=tokens_to_idxs, parallel=parallel, num_workers=args_phase3.num_workers)

    # DataLoader flags for phase 3
    use_cuda = (args_phase3.device == "cuda")
    use_persist = args_phase3.num_workers > 0

    tr_loader_tri = torch.utils.data.DataLoader(
        train_set_tri, batch_size=args_phase3.batch_size, collate_fn=collate_fn,
        shuffle=True, pin_memory=use_cuda,
        num_workers=args_phase3.num_workers,
        persistent_workers=use_persist)
    va_loader_tri = torch.utils.data.DataLoader(
        valid_set_tri, batch_size=args_phase3.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda,
        num_workers=args_phase3.num_workers,
        persistent_workers=use_persist)
    te_loader_tri = torch.utils.data.DataLoader(
        test_set_tri, batch_size=args_phase3.val_batch_size, collate_fn=collate_fn,
        shuffle=False, pin_memory=use_cuda,
        num_workers=min(args_phase3.num_workers, MAX_WORKERS_TEST),
        persistent_workers=False)

    model_tri = PathEModelTriples(
        vocab_size=train_set_tri.vocab_size,
        padding_idx=tokens_to_idxs["PAD"],
        relcontext_graph=relcontext_graph,
        **bundle(target_class=PathEModelTriples),
    )
    pl_model_tri = PathEModelWrapperTriples(
        pathe_model=model_tri,
        filtration_dict=filtration_dict,
        class_weights=class_weights,
        **namespace_to_dict(args_phase3),
    )
    pl_model_tri.model.to(torch.device(args_phase3.device))

    # --- Provide per group positives count to recall metrics
    def count_positives_per_group(triples: torch.Tensor, get_group_ids_fn: Callable[[torch.Tensor], torch.Tensor]) -> dict[int, int]:
        group_ids = get_group_ids_fn(triples).to(torch.long)
        if group_ids.numel() == 0:
            return {}
        counts = torch.bincount(group_ids, minlength=int(group_ids.max().item()) + 1)
        # keep only non-zero counts if perhaps group_ids where not consecutive starting with 0
        nonzero = (counts > 0).nonzero(as_tuple=False).flatten()
        return {int(i): int(counts[i].item()) for i in nonzero}

    # positives per-group counts for validation/test (same grouping as dataset)
    val_true_counts = count_positives_per_group(val_triples, get_group_ids)
    test_true_counts = count_positives_per_group(test_triples, get_group_ids)

    # Inject counts into per-group recall metrics
    for k in pl_model_tri.cand_topk:
        pl_model_tri.cand_metrics_val[f"recall@{k}_perGroup"].set_true_counts(val_true_counts)
        pl_model_tri.cand_metrics_val[f"recall@{k}_total"].set_num_positives(len(val_triples))
        pl_model_tri.cand_metrics_test[f"recall@{k}_perGroup"].set_true_counts(test_true_counts)
        pl_model_tri.cand_metrics_test[f"recall@{k}_total"].set_num_positives(len(test_triples))

    # Loggers and trainer
    tb_logger_tri = TensorBoardLogger(save_dir=args_phase3.log_dir, name=args_phase3.expname, version=args_phase3.version, sub_dir="triples")
    if args_phase3.wandb_project is not None:
        wb_logger_tri = WandbLogger(id=args_phase3.wandb_id, save_dir=tb_logger_tri.log_dir,
                                    name=f"{args_phase3.expname}_triples", project=args_phase3.wandb_project,
                                    log_model="all", sync_tensorboard=True)
        wb_logger_tri.watch(pl_model_tri, log="all"); wb_logger_tri.log_hyperparams(args_phase3)
    else:
        wb_logger_tri = None
        tb_logger_tri.log_hyperparams(args_phase3)

    mode = "min" if args_phase3.triple_monitor.endswith("loss") else "max"
    checkpoint_callbk_tri = ModelCheckpoint(
        monitor=args_phase3.triple_monitor, dirpath=args_phase3.checkpoint_dir, mode=mode,
        filename=model_name + f"-triple-{{epoch}}-{{{args_phase3.triple_monitor}:.2f}}",
        every_n_train_steps=args_phase3.chekpoint_ksteps)
    estopping_callbk_tri = EarlyStopping(
        monitor=args_phase3.triple_monitor, patience=args_phase3.patience, mode=mode, check_on_train_epoch_end=False)
    dataset_callbk_tri = DatasetUpdater([train_set_tri, valid_set_tri] if args_phase3.train_paths else [train_set_tri.dataset])

    trainer_tri = Trainer(
        max_epochs=args_phase3.max_epochs, gradient_clip_val=1.0,
        accelerator=accelerator, devices=args_phase3.num_devices, num_nodes=1,
        limit_train_batches=tr_limit, limit_val_batches=va_limit,
        logger=[tb_logger_tri] + ([wb_logger_tri] if wb_logger_tri else []),
        log_every_n_steps=5, val_check_interval=args_phase3.val_check_interval,
        check_val_every_n_epoch=args_phase3.check_val_every_n_epoch,
        gradient_clip_algorithm='norm', accumulate_grad_batches=args_phase3.accumulate_gradient,
        callbacks=[estopping_callbk_tri, checkpoint_callbk_tri, dataset_callbk_tri],
        num_sanity_val_steps=-1,
    )

    # NEW: Evaluate triple metrics on candidate test set with untrained model (baseline/random performance)
    stageprint("Evaluating untrained triple model on candidate test set (baseline metrics)...")
    untrained_test_dict = trainer_tri.test(
        model=pl_model_tri,  # Use the untrained model
        dataloaders=te_loader_tri,  # Candidate test set
        ckpt_path=None  # No checkpoint; use current (untrained) model state
    )[0]
    print("\nUntrained triple model results on candidates: {}".format(untrained_test_dict))


    if args_phase3.cmd in ["train", "resume"] and not args_phase3.skip_phase2:
        stageprint("Training-validating the model, be patient!")
        if args_phase3.cmd == "resume":
            # Run validation first to log metrics (e.g., 'valid_link_mrr') before training starts
            trainer_tri.validate(pl_model_tri, va_loader_tri, ckpt_path=args_phase3.triple_checkpoint)
        trainer_tri.fit(pl_model_tri, tr_loader_tri, va_loader_tri, ckpt_path=args_phase3.triple_checkpoint)
        triple_ckpt = checkpoint_callbk_tri.best_model_path
        print(f"[Triples] Done. Best model saved in {triple_ckpt}")
        ttime = (datetime.datetime.now() - start_phase3_time).total_seconds() / 3600
        print(f"[Triples] Training time: {round(ttime, 2)} hours")
    else:
        triple_ckpt = args_phase3.triple_checkpoint
        print(f"[Triples] Using checkpoint for test: {triple_ckpt}")
        if args_phase3.cmd == "test" and not triple_ckpt:
            raise ValueError("triple_checkpoint is required for cmd='test'.")

    if triple_ckpt:
        stageprint("Evaluating the model on the test set")
        test_dict = trainer_tri.test(
            model=pl_model_tri if args_phase3.cmd == "test" else None,
            dataloaders=te_loader_tri,
            ckpt_path=triple_ckpt
        )[0]
        print("\nFinal testing results (triple model): {}".format(test_dict))

    # Cleanup before exit
    print("Cleaning up resources...")
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print("Done! You can enter the next command.")