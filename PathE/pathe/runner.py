"""
PathE: main running entry-point for training, evaluating and testing.

"""
import os
import joblib
import argparse

import torch

from . import pathe_trainer
from .utils import is_file, create_dir, set_random_seed
from .utils import resume_logging_setup, create_logging_setup, update_args
from .pathe_full_eval import run_full_eval


def main():
    """
    Main function to parse the arguments and call the main process.
    """

    parser = argparse.ArgumentParser(
        description='Main entry point for PathE | Trainer, runner, tester.')

    parser.add_argument('cmd', choices=['train', 'resume', 'test', 'full_eval'],
                        help='A supported command to execute.')
    parser.add_argument('model', choices=['pathe', 'patheTuples', 'pathe2Phases'],
                        help='The name of the PathE model to use.')
    parser.add_argument('--path_type', choices=['unrolled'],
                        default='unrolled', help='unrolled')
    parser.add_argument('--path_setup', type=str, default='20_20',
                        help='The path configuration to load if more present')
    parser.add_argument('--expname', type=str, default="pathe",
                        help='The name of the experiment/run to record.')
    parser.add_argument('--pathstore', type=lambda x: is_file(parser, x),
                        help='Path to the data bundle with data and encoders.')
    parser.add_argument('--train_paths', type=lambda x: is_file(parser, x),
                        help='Path to the file containing training path data.')
    parser.add_argument('--valid_paths', type=lambda x: is_file(parser, x),
                        help='Path to the file containing validation path data.')
    parser.add_argument('--test_paths', type=lambda x: is_file(parser, x),
                        help='Path to the file containing test path data.')

    parser.add_argument('--assignments', type=lambda x: is_file(parser, x),
                        help='Partition assignment for each path in pathstore.')
    parser.add_argument('--max_ppt', action='store', type=int, default=50,
                        help='Maximum number of paths per triple to sample.')
    parser.add_argument('--augmentation_factor', action='store', type=int, default=20,
                        help='Factor for data augmentation during training.')
    parser.add_argument('--xtokens', type=float, nargs="+", default=["MSK"],
                        help='All the special tokens that will be embedded.')

    # Model hyper-parameters and defaults
    # The next hparameters control the aggregation strategy for entities
    parser.add_argument('--ent_aggregation', action='store',
                        choices=['avg', "recurrent", "transformer", "masked_mean"], default='avg',
                        help='The name of the entity aggregation strategy.')
    parser.add_argument('--num_agg_heads', action='store', type=int, default=1,
                        help='Number of attention heads in the aggregator. '
                             'Only used when ent_aggregation is based on SAN.')
    parser.add_argument('--num_agg_layers', action='store', type=int, default=1,
                        help='Number of layers in the aggregator.')
    parser.add_argument('--context_heads', action='store', type=int, default=0,
                        help='Number of heads for entity contextualisation.'
                             'Only enabled when greater than 0. Default 0.')
    # The next hyperparameters control the transformer encoder of PathE
    parser.add_argument('--embedding_dim', dest="d_model", type=int, default=64,
                        help='Size of the embedding vectors to learn.')
    parser.add_argument('--nhead', action='store', type=int, default=8,
                        help='Number of attention heads in the encoder.')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers in the arhictecture.')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Dimension of the feedforward network model.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='The dropout percentage at training time.')
    parser.add_argument('--train_cls', action="store_true", default=False,
                        help='Whether to train the CLS token on rel prediction.')
    parser.add_argument('--max_seqlen', action='store', type=int, default=100,
                        help='Maximum length of entity-relation paths.')
    
    # Parameters for candidate triple generation and filtering
    parser.add_argument('--candidates_threshold_p', type=float, default=None,
                        help='Global probability threshold for candidate triples (keep those with P >= p).')
    parser.add_argument('--candidates_quantile_q', type=float, default=None,
                        help='Global quantile threshold for candidate triples (keep top (1-q) quantile).')
    parser.add_argument('--candidates_temperature', type=float, default=1.0,
                        help='Temperature for candidate probability calibration.')
    parser.add_argument('--candidates_alpha', type=float, default=0.5,
                        help='Weight for tail vs head in candidate scoring (0=head only, 1=tail only).')
    parser.add_argument('--candidates_beta', type=float, default=0.5,
                        help='Weight for tail prediction probability in candidate scoring. Only used if candidate_generator includes tail predictions.')
    parser.add_argument('--candidates_cap', type=int, default=100,
                        help='Maximum number of top-k candidates to keep per group (e.g., per head entity).')
    parser.add_argument('--candidates_normalize_mode', choices=['per_head', 'global_joint', 'per_relation', 'none'], default='global_joint',
                        help='Normalization mode for candidate scoring: per_head (conditional probs), global_joint (joint probs), per_relation (normalize per relation), none (raw logits).')
    parser.add_argument('--candidate_generator', choices=['global', 'global_with_tail', 'per_head'], default='global_with_tail',
                        help='Type of candidate generator to use: global (threshold/quantile-based), global_with_tail (includes tail logits), per_head (top-k per head).')
    parser.add_argument('--group_strategy', type=int, nargs='+', default=[0],
                        help='Columns to group by for candidate generation (e.g., [0] for head entities).')
    parser.add_argument('--figure_dir', action='store', default="./figures",
                        help='Directory where figures will be saved.')

    # Logging and checkpointing
    parser.add_argument('--log_dir', action='store', default="experiments",
                        help='Directory where log files will be generated.')
    parser.add_argument('--version', action='store', type=int, default=None,
                        help='Version number for this experiment to resume.')
    parser.add_argument('--wandb_project', action='store', type=str,
                        help='The name of the wandb project for logging.')
    parser.add_argument('--checkpointing', action='store_true', default=False,
                        help='Whether the model state dict will be dumped.')
    parser.add_argument('--triple_checkpoint', type=lambda x: is_file(parser, x),
                        help='Path to a triple model checkpoint to load.')
    parser.add_argument('--tuple_checkpoint', type=lambda x: is_file(parser, x),
                        help='Path to a tuple model checkpoint to load.')

    # The following arguments are used to control the training process
    parser.add_argument('--lp_loss_fn', action='store',
                        choices=['bce', 'ce', 'nssa'], default='nssa',
                        help='The name of the loss to use for link prediction.')
    parser.add_argument('--phase1_loss_fn', action='store',
                        choices=['bce', 'poisson', 'hurdletail', 'hurdlerelation', 'hurdleboth', 'negative_binomial'], default='bce',
                        help='The name of the loss to use for phase 1 (tuples) training: bce or poisson.')
    parser.add_argument('--use_manual_optimization', action='store_true', default=False,
                        help='Whether to use manual optimization with independent heads.')
    parser.add_argument('--link_head_detached', action='store_true', default=False,
                        help='Whether to detach the link head during training (used for best relation prediction results).')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Whether to resume training of the model.')
    parser.add_argument('--batch_size', action='store', type=int, default=16,
                        help='Batch size for training and validation loaders.')
    parser.add_argument('--lrate', action='store', type=float, default=1e-3,
                        help='Starting learning rate (before any scheduling).')
    parser.add_argument('--scheduler', action='store', choices=['none', 'reduce_on_plateau'], default='none',
                        help='The learning rate scheduler to use.')
    parser.add_argument('--scheduler_monitor', action='store', default="valid_total_loss",
                        help='The metric to monitor for the scheduler.')
    parser.add_argument('--scheduler_patience', action='store', type=int, default=5,
                        help='Patience for the scheduler.')
    parser.add_argument('--loss_weight', type=float, default=0.5,
                        help='Controls loss weighting for multi-task learning. '
                             'The weight for the relation prediction loss is '
                             'implemented as (1-loss_weight). Must be [0,1]')
    parser.add_argument('--weight_decay', action='store', type=float, default=0,
                        help='Weight decay for L2 regularisation.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='The amount of smoothing when computing the loss.')
    parser.add_argument('--max_epochs', action='store', type=int, default=10000,
                        help='Maximum number of training epochs to run.')
    parser.add_argument('--patience', action='store', type=int, default=10,
                        help='Number of validation epochs with no improvement.')
    parser.add_argument('--tuple_monitor', action='store', default="valid_mrr",
                        choices=["valid_rp_loss", "valid_tp_loss", "valid_total_loss", "valid_mrr", "valid_link_mrr",
                                 "valid_tail_mrr", "valid_relation_rmse", "valid_tail_rmse",
                                 "valid_link_hits@1", "valid_link_hits@3", "valid_link_hits@5", "valid_link_hits@10"],
                        help='Monitored metric for early stopping and ckpt for tuples. '
                             'For counting loss functions (poisson, negative_binomial, etc.), use rmse metrics.')
    parser.add_argument('--triple_monitor', action='store', default="valid_link_mrr",
                        choices=["valid_rp_loss", "valid_lp_loss", "valid_total_loss", "valid_mrr", "valid_link_mrr", 
                                 "valid_link_hits@1", "valid_link_hits@3", "valid_link_hits@5", "valid_link_hits@10", 
                                 "valid_link_recall@5_perGroup", "valid_link_recall@10_perGroup"],
                        help='Monitored metric for early stopping and ckpt for triples.')
    parser.add_argument('--class_weigths', action='store_true', default=False,
                        help='Whether to weight the loss with class frequencies.')
    parser.add_argument('--accumulate_gradient', type=int, default=1,
                        help='No. of batches for gradient accumulation (1==off).')

    parser.add_argument('--dtype', action='store', choices=['d', 'f'], default='f',
                        help='Data type of tensors. Default: f (float).')
    parser.add_argument('--device', action='store', default='cpu',
                        help='Device to use for training and validation.')
    parser.add_argument('--num_devices', action='store', type=int, default=None,
                        help='Either CPU threads or GPU units to allocate.')
    parser.add_argument('--num_workers', action='store', type=int, default=0,
                        help='No. of threads for parallel data loading.')
    parser.add_argument('--seed', action='store', type=int, default=42,
                        help='Random seed for reproducibility of the exp.')
    parser.add_argument('--multigpu', action='store_true', default=False,
                        help='Whether to use all the GPU devices available.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Whether to run a small-scale experiment.')
    parser.add_argument('--skip_phase1', action='store_true', default=False,
                        help='Whether to skip training the first phase (triples). Has no effect if model is not pathe2Phases.')
    parser.add_argument('--skip_phase2', action='store_true', default=False,
                        help='Whether to skip training the second phase (candidates). Has no effect if model is not pathe2Phases.')


    # Parameters for the evaluation data and strategies
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help='The batch size for the val/test data loaders.')
    parser.add_argument('--train_sub_batch', type=int, default=None,
                        help='The sub-batch size for the training set.')
    parser.add_argument('--val_sub_batch', type=int, default=None,
                        help='The sub-batch size for the validation set.')
    parser.add_argument('--test_sub_batch', type=int, default=None,
                        help='The sub-batch size for the test set.')
    parser.add_argument('--val_relcontext_size', type=int, default=5,
                        help='The size of the relational context to sample.')
    parser.add_argument('--num_negatives', type=int, default=0,
                        help='Number of negative/corrupt triples per positive.')
    parser.add_argument('--val_num_negatives', type=int, default=None,
                        help='No. of negative/corrupt triples per positive '
                             'for validation and test sets.')
    parser.add_argument('--full_test', action='store_true', default=False,
                        help='Whether running a full test with all negatives. '
                             'This overrides val_num_negatives for test set.')
    parser.add_argument('--corruption', choices=['h', 't', 'all', 'r', 'e'], default="all",
                        help='Triple corruption strategy for link prediction; '
                             'head (h), tail (t), or both merged (all).')
    parser.add_argument('--dump_negatives', action='store_true', default=False,
                        help='Whether to dump the negative triples in expdir.')
    parser.add_argument('--dump_dir', action='store', type=str,
                        help='If provided, data will be loaded from the dump.')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='How often per epoch to check the validation set.')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Number of epochs between validation checks.')
    parser.add_argument('--chekpoint_ksteps', type=int, default=None,
                        help='If given, dumps the model every k train steps.')

    # Ablation parameters
    parser.add_argument('--node_projector', choices=[ "dummy"], default="dummy",
                        help='One of the supported types of node projector.')
    parser.add_argument('--simple_positional', action='store_true', default=False,
                        help='Whether to use a simple positional encodings.')

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, args.model)
    args.figure_dir = os.path.join(args.figure_dir, args.model, args.expname, args.candidate_generator, args.candidates_normalize_mode)
    # Filling missing/optional values if not provided ...
    # args.dtype = torch.double if args.dtype == 'd' else torch.float FIXME
    if not (0 <= args.loss_weight <= 1):
        raise ValueError("Loss_weight must be between [0,1]")
    # args.device = torch.device(args.device)
    args.val_batch_size = args.batch_size if args.val_batch_size is None else args.val_batch_size
    args.val_num_negatives = args.num_negatives if args.val_num_negatives is None else args.val_num_negatives
    if args.link_head_detached and not args.use_manual_optimization:
        raise ValueError("link_head_detached=True has no effect when use_manual_optimization=False")
    # Warning for scheduler patience vs. early stopping patience
    if args.scheduler != 'none' and args.scheduler_patience >= args.patience:
        print(f"Warning: Scheduler patience ({args.scheduler_patience}) is not smaller than early stopping patience ({args.patience}). "
            "Consider reducing --scheduler_patience to allow LR reduction before early stopping.")


    # Setting the random seed for all modules
    set_random_seed(args.seed, args.device)

    # create_dir(args.log_dir)  # root folder for the experiment
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    print(f"Model: {args.model} to train on {args.path_type} paths")
    print(f"Using {args.device} (tensor type: {args.dtype}) with random"
          f" seed {args.seed} | Logging in: {args.log_dir}")
    
    # dump_configuration(args, logdir="./")
    # new_args = load_configuration(logdir="./")

    if args.cmd in ["resume", "test"]:
        print("RESUME/TEST mode: retrieving experimental setup")
        update_args(args, resume_logging_setup(
            args.log_dir, args.expname, args.version, args.wandb_project))
    elif args.cmd == "train":
        print("TRAIN mode: creating new experimental setup")
        update_args(args, create_logging_setup(
            args.log_dir, args.expname, args.version))
    elif args.cmd == "full_eval":
        print("Full EVAL mode: Loading model for evaluation")
        run_full_eval(args)
        return


    # print(args)
    if args.model == "pathe":
        if args.val_num_negatives == 0:
            assert "link" not in args.triple_monitor, f"Link prediction metric {args.triple_monitor} cannot be used when val_num_negatives=0"
        assert "recall" not in args.triple_monitor, f"Recall metrics cannot be used with single phase triple model."
        if args.loss_weight != 1:
            print("Warning: loss_weight != 1 does also train rp in triples mode, this may be unintended.")
        pathe_trainer.create_and_run_training_exp_triples(args)
    elif args.model == "patheTuples":
        if args.val_num_negatives == 0:
            assert "link" not in args.tuple_monitor, f"Link prediction metric {args.tuple_monitor} cannot be used when val_num_negatives=0"
        pathe_trainer.create_and_run_training_exp_tuples(args)
    elif args.model == "pathe2Phases":
        if args.val_num_negatives == 0:
            assert "link" not in args.tuple_monitor, f"Link prediction metric {args.tuple_monitor} cannot be used when val_num_negatives=0"
        assert args.use_manual_optimization, "Two-phase training requires --use_manual_optimization to be set for proper relation prediction in tuples training."
        # assert args.link_head_detached, "Two-phase training requires --link_head_detached to be set for proper relation prediction in tuples training."
        assert not (args.tuple_checkpoint is None and args.skip_phase1), "Cannot skip phase 1 if no tuple_checkpoint is provided."
        pathe_trainer.create_and_run_training_exp_two_phases(args)



if __name__ == "__main__":
    main()
