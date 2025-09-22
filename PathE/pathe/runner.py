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
    parser.add_argument('--xtokens', type=float, nargs="+", default=["MSK"],
                        help='All the special tokens that will be embedded.')

    # Model hyper-parameters and defaults
    # The next hparameters control the aggregation strategy for entities
    parser.add_argument('--ent_aggregation', action='store',
                        choices=['avg', "recurrent", "transformer"], default='avg',
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
    
    parser.add_argument('--candidates_topk', type=int, default=1000000,
                        help='Number of top relation candidates per entity to use for triple candidate generation in 2Phase model.')

    # Logging and checkpointing
    parser.add_argument('--log_dir', action='store', default="experiments",
                        help='Directory where log files will be generated.')
    parser.add_argument('--version', action='store', type=int, default=None,
                        help='Version number for this experiment to resume.')
    parser.add_argument('--wandb_project', action='store', type=str,
                        help='The name of the wandb project for logging.')
    parser.add_argument('--checkpointing', action='store_true', default=False,
                        help='Whether the model state dict will be dumped.')
    parser.add_argument('--checkpoint', type=lambda x: is_file(parser, x),
                        help='Path to a model checkpoint to load. Use only if single phase model is used.')
    parser.add_argument('--triple_checkpoint', type=lambda x: is_file(parser, x),
                        help='Path to a triple model checkpoint to load. Use only if two-phase model is used.')
    parser.add_argument('--tuple_checkpoint', type=lambda x: is_file(parser, x),
                        help='Path to a tuple model checkpoint to load. Use only if two-phase model is used.')

    # The following arguments are used to control the training process
    parser.add_argument('--lp_loss_fn', action='store',
                        choices=['bce', 'ce', 'nssa'], default='nssa',
                        help='The name of the loss to use for link prediction.')
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
    parser.add_argument('--loss_weight', type=float, default=1.,
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
                        choices=["valid_rp_loss", "valid_lp_loss", "valid_total_loss", "valid_mrr", "valid_link_mrr",
                                 "valid_link_hits@1", "valid_link_hits@3", "valid_link_hits@5", "valid_link_hits@10"],
                        help='Monitored metric for early stopping and ckpt for tuples.')
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
    parser.add_argument('--seed', action='store', type=int, default=46,
                        help='Random seed for reproducibility of the exp.')
    parser.add_argument('--multigpu', action='store_true', default=False,
                        help='Whether to use all the GPU devices available.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Whether to run a small-scale experiment.')


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
    parser.add_argument('--chekpoint_ksteps', type=int, default=None,
                        help='If given, dumps the model every k train steps.')

    # Ablation parameters
    parser.add_argument('--node_projector', choices=[ "dummy"], default="dummy",
                        help='One of the supported types of node projector.')
    parser.add_argument('--simple_positional', action='store_true', default=False,
                        help='Whether to use a simple positional encodings.')

    args = parser.parse_args()
    # Filling missing/optional values if not provided ...
    # args.dtype = torch.double if args.dtype == 'd' else torch.float FIXME
    if not (0 <= args.loss_weight <= 1):
        raise ValueError("Loss_weight must be between [0,1]")
    # args.device = torch.device(args.device)
    args.val_batch_size = args.batch_size if args.val_batch_size is None else args.val_batch_size
    args.val_num_negatives = args.num_negatives if args.val_num_negatives is None else args.val_num_negatives
    if args.link_head_detached and not args.use_manual_optimization:
        raise ValueError("link_head_detached=True has no effect when use_manual_optimization=False")

    # Setting the random seed for all modules
    set_random_seed(args.seed, args.device)

    # create_dir(args.log_dir)  # root folder for the experiment
    os.makedirs(args.log_dir, exist_ok=True)

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
        assert args.triple_checkpoint is None, "triple_checkpoint can only be used with pathe2Phases"
        if args.val_num_negatives == 0:
            assert "link" not in args.triple_monitor, f"Link prediction metric {args.triple_monitor} cannot be used when val_num_negatives=0"
        assert "recall" not in args.triple_monitor, f"Recall metrics cannot be used with single phase triple model."
        pathe_trainer.create_and_run_training_exp_triples(args)
    elif args.model == "patheTuples":
        if args.val_num_negatives == 0:
            assert "link" not in args.tuple_monitor, f"Link prediction metric {args.tuple_monitor} cannot be used when val_num_negatives=0"
        assert args.tuple_checkpoint is None, "tuple_checkpoint can only be used with pathe2Phases"
        pathe_trainer.create_and_run_training_exp_tuples(args)
    elif args.model == "pathe2Phases":
        if args.val_num_negatives == 0:
            assert "link" not in args.tuple_monitor, f"Link prediction metric {args.tuple_monitor} cannot be used when val_num_negatives=0"
        assert args.checkpoint is None, "checkpoint cannot be used with pathe2Phases. Use triple_checkpoint and/or tuple_checkpoint instead."
        assert args.use_manual_optimization, "Two-phase training requires --use_manual_optimization to be set for proper relation prediction in tuples training."
        assert args.link_head_detached, "Two-phase training requires --link_head_detached to be set for proper relation prediction in tuples training."
        pathe_trainer.create_and_run_training_exp_two_phases(args)



if __name__ == "__main__":
    main()
