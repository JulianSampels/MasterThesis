"""
Scalable routines for training and finetuning a PathER model.

"""

import logging
from functools import partial
import math
from typing import Optional

import torch
from torch import nn
import pandas as pd
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer
import torch_scatter

from PathE.pathe.pathdata import RelationMaps

from .pathe_ranking_metrics import (RelationMRRTriples, RelationMRRTuples, RelationHitsAtKTriples, RelationHitsAtKTuples,
                                   EntityMRRTriples, EntityHitsAtKTriples, CandidateMRRPerSampleFiltered, 
                                   CandidateHitsAtKPerSampleFiltered, CandidateRecallAtKPerGroup)

from .pather_models import PathEModelTriples, PathEModelTuples


logger = logging.getLogger(__name__)


def debug_single_path(batch, trim_paths: bool=False, override_pos: bool=False):
    if trim_paths:
        path_origins = batch['net_input']['path_origins']
        new_path_origins, counts = torch.unique_consecutive(path_origins,
                                                            return_counts=True)
        ccounts = torch.cumsum(counts, dim=0)
        ccounts_clone = ccounts.clone()
        ccounts[1:] = ccounts_clone[:-1]
        ccounts[0] = 0
        new_ppt = torch.ones_like(batch["ppt"]) * 2
        new_ent_paths = batch['net_input']['ent_paths'][ccounts, :]
        new_rel_paths = batch['net_input']['rel_paths'][ccounts, :]
        new_ent_lengths = batch['net_input']['ent_lengths'][ccounts]
        new_head_idxs = batch['net_input']['head_idxs'][ccounts]
        new_tail_idxs = batch['net_input']['tail_idxs'][ccounts]
        new_pos = batch['net_input']['pos'][ccounts, :]

        batch['ppt'] = new_ppt
        batch['net_input']['ent_paths'] = new_ent_paths
        batch['net_input']['rel_paths'] = new_rel_paths
        batch['net_input']['ent_lengths'] = new_ent_lengths
        batch['net_input']['head_idxs'] = new_head_idxs
        batch['net_input']['tail_idxs'] = new_tail_idxs
        batch['net_input']['pos'] = new_pos
        batch['net_input']['path_origins'] = new_path_origins
    if override_pos:
        device = batch['net_input']['pos'].device
        pos_size = batch['net_input']['pos'].size()
        new_pos = torch.arange(1, pos_size[1]+1, device=device).repeat(
            pos_size[0], 1)
        new_pos[(batch['net_input']['pos'] == 0)] = 0
        previous = batch['net_input']['pos']
        batch['net_input']['pos'] = new_pos
        del(previous)


    return batch



def pathe_forward_step(model, criterion, batch, **kwargs):
    """
    PathE forward function for multi-paths relation prediction.
    """
    # batch = debug_single_path(batch, trim_paths=False, override_pos=True)
    ent_paths = batch["net_input"]["ent_paths"]
    rel_paths = batch["net_input"]["rel_paths"]
    head_idxs = batch["net_input"]["head_idxs"]
    tail_idxs = batch["net_input"]["tail_idxs"]
    entity_origin = batch["net_input"]["path_origins"]
    pos = batch["net_input"]["pos"]
    targets = batch["target"] - 2  # no PAD and MSK in the model output
    # FIXME this is going to change with the actual head-tail idxs from dataset
    head_idxs = head_idxs * 2
    tail_idxs = tail_idxs * 2
    ppt = batch["ppt"]

    logits_rp, logits_lp = model(
        ent_paths=ent_paths,
        rel_paths=rel_paths,
        head_idxs=head_idxs,
        tail_idxs=tail_idxs,
        ppt=ppt, pos=pos,
        entity_origin=entity_origin,
        targets=targets)
    # loss_rp = criterion(logits_rp, batch["target"])
    loss_rp = criterion(logits_rp, targets)
    return logits_rp, loss_rp, logits_lp


class PathEModelWrapperTriples(LightningModule):
    """
    PL Wrapper for training PathE for triples.

    """

    def __init__(self, pathe_model, filtration_dict, num_negatives=0,
                 optimiser="adam", scheduler="none", lr=1e-3, momentum=0,
                 weight_decay=0, class_weights=None, label_smoothing=0.0,
                 train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                 val_num_negatives=0, full_test=False, max_ppt=None,
                 margin=10, nssa_alpha=1, lp_loss_fn="nssa",
                 loss_weight: float = 0.5,  **hparams):
        """
        Parameters
        ----------
        pather_model : torch.Module
            A supported Seq2Seq model, with encoder and decoder (e.g. ParthX).
        tokens_to_idxs : dict
            The mapping/encoding from original tokens to their indexes, needed
            for the extraction of the metrics that will be monitored.
        train_cls : bool
            Whether to train the CLS objective on the relation prediction task.

        """
        super().__init__()
        self.model = pathe_model
        self.loss_weight = loss_weight
        self.train_num_negatives = num_negatives
        self.val_num_negatives = val_num_negatives
        self.test_num_negatives = val_num_negatives if not full_test else full_test
        self.class_weights = class_weights
        comp_sub_batch = lambda sub_batch: sub_batch * max_ppt \
            if sub_batch is not None else sub_batch  # CHANGED THIS
        self.train_sub_batch_size = comp_sub_batch(train_sub_batch)
        self.val_sub_batch_size = comp_sub_batch(val_sub_batch) 
        self.test_sub_batch_size = comp_sub_batch(test_sub_batch)
        # Optimisation hyperparams
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.lr = lr  # initial LR
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        # [List of metrics we will be watching on validation and test sets]
        # Relation prediction metrics: always watched
        self.val_relationMRR = RelationMRRTriples(filtration_dict)
        self.val_relationHitsAt1 = RelationHitsAtKTriples(filtration_dict, k=1)
        self.val_relationHitsAt3 = RelationHitsAtKTriples(filtration_dict, k=3)
        self.val_relationHitsAt5 = RelationHitsAtKTriples(filtration_dict, k=5)
        self.val_relationHitsAt10 = RelationHitsAtKTriples(filtration_dict, k=10)
        self.test_relationMRR = RelationMRRTriples(filtration_dict)
        self.test_relationHitsAt1 = RelationHitsAtKTriples(filtration_dict, k=1)
        self.test_relationHitsAt3 = RelationHitsAtKTriples(filtration_dict, k=3)
        self.test_relationHitsAt5 = RelationHitsAtKTriples(filtration_dict, k=5)
        self.test_relationHitsAt10 = RelationHitsAtKTriples(filtration_dict, k=10)
        # self.save_hyperparameters(ignore=['pathe_model'])

        if self.val_num_negatives > 0:  # watch link prediction metrics
            self.val_linkMRR = EntityMRRTriples()
            self.val_linkHitsAt1 = EntityHitsAtKTriples(k=1)
            self.val_linkHitsAt3 = EntityHitsAtKTriples(k=3)
            self.val_linkHitsAt5 = EntityHitsAtKTriples(k=5)
            self.val_linkHitsAt10 = EntityHitsAtKTriples(k=10)
            self.test_linkMRR = EntityMRRTriples()
            self.test_linkHitsAt1 = EntityHitsAtKTriples(k=1)
            self.test_linkHitsAt3 = EntityHitsAtKTriples(k=3)
            self.test_linkHitsAt5 = EntityHitsAtKTriples(k=5)
            self.test_linkHitsAt10 = EntityHitsAtKTriples(k=10)
        
        # Candidate metrics (torchmetric style)
        self.cand_topk = (1, 3, 5, 10)
        self.cand_metrics_val = nn.ModuleDict({
            "mrr": CandidateMRRPerSampleFiltered(),
            **{f"hits@{k}": CandidateHitsAtKPerSampleFiltered(k) for k in self.cand_topk},
            **{f"recall@{k}_perGroup": CandidateRecallAtKPerGroup(k) for k in self.cand_topk},
        })
        self.cand_metrics_test = nn.ModuleDict({
            "mrr": CandidateMRRPerSampleFiltered(),
            **{f"hits@{k}": CandidateHitsAtKPerSampleFiltered(k) for k in self.cand_topk},
            **{f"recall@{k}_perGroup": CandidateRecallAtKPerGroup(k) for k in self.cand_topk},
        })

        # Losses
        # self.rp_criterion = torch.nn.MultiMarginLoss(weight=class_weights, margin=margin)

        self.rp_loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction="none",
            label_smoothing=label_smoothing)
        
        if lp_loss_fn == "bce":
            self.lp_loss_fn = self.calculate_lp_bce
        elif lp_loss_fn == "ce":
            self.lp_loss_fn = self.calculate_lp_ce
        elif lp_loss_fn == "nssa":
            self.lp_loss_fn = partial(
                self.calculate_lp_nssa, alpha=nssa_alpha, gamma=margin)
        else:
            raise ValueError(f"Unknown lp_loss_fn: {lp_loss_fn}")

    def calculate_lp_bce(self, logits, num_negatives = None, labels: torch.Tensor = None, sample_weights: torch.Tensor = None):
        """
        Calculates the weighted BCE loss for link prediction.
        Each negative triple is weighted by 1/num negatives.

        Parameters
        ----------
        logits : The logits for each triple with shape (num_triples,1)
        labels : The labels for each triple with shape (num_triples,), optional
            If provided, the loss is calculated as weighted BCE with these labels.
        sample_weights : The weights for each triple with shape (num_triples,), optional
            If provided, the loss is calculated as weighted BCE with these weights.
        num_negatives : The number of negatives, optional
            If provided, the loss is calculated in legacy mode with num_negatives.

        """
        assert not (labels is not None and sample_weights is not None) or (not num_negatives), "If labels and sample_weights are provided, num_negatives must be 0 or None!"
        assert (labels is None) == (sample_weights is None), "labels and sample_weights must be both provided or both None!"
        # candidate mode with provided labels and weights
        if labels is not None and sample_weights is not None:
            # Weighted mean BCE loss with provided labels and weights
            logits = logits.squeeze(-1)
            labels = labels.to(logits.dtype)
            w = sample_weights.to(logits.dtype)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=w, reduction="sum")
            return loss / w.sum().clamp_min(1.0)
        
        # Legacy mode with num_negatives
        assert num_negatives is not None, "Either labels and sample_weights or num_negatives must be provided!"
        # create the labels for the triples as all zeros
        labels = torch.zeros((logits.size()[0]), dtype=torch.float32).to(self.device)
        # make the labels of the true triples 1
        labels[torch.arange(0, labels.size()[0], num_negatives + 1, device=labels.device)] = 1.
        # calculate the weight matrix
        downweigh_negs: torch.Tensor = torch.ones_like(labels) / num_negatives
        # set the weights of true triple to 1
        downweigh_negs[torch.arange(0, labels.size()[0], num_negatives + 1, device=downweigh_negs.device)] = 1.
        loss = nn.functional.binary_cross_entropy_with_logits(logits.squeeze(), labels, reduction="none")
        loss = (loss * downweigh_negs).sum(dim=-1) / downweigh_negs.sum(dim=-1)
        return loss

    def calculate_lp_ce(self, logits, num_negatives):
        """
        Calculates the CE loss for link prediction.

        Parameters
        ----------
        logits : The logits for each triple with shape (num_triples,1)
        num_negatives : The number of negatives
    
        """
        scores_unpacked = logits.reshape(logits.size()[0] //
                                         (num_negatives + 1),
                                         num_negatives + 1)
        labels = torch.zeros((scores_unpacked.size()[0]), dtype=torch.int64, device=self.device)
        loss_lp = torch.mean(nn.functional.cross_entropy(scores_unpacked, labels))
        return loss_lp

    def calculate_lp_nssa(self, scores, num_negs, alpha=1, gamma=10):
        """
        Self-Adversarial Negative Sampling Loss as shown in Sun et al. (RotatE)
        
        Parameters
        ----------
        scores : The logits of the model (num samples, 1)
        num_negs : The number of negatives for each positive sample
        alpha : The adversarial temperature
        gamma : The margin

        """
        score = gamma + scores
        scores_per_sample = num_negs + 1
        unstacked_scores = score.reshape(
            score.size()[0] // scores_per_sample, scores_per_sample, -1)
        
        pos_score = score[torch.arange(0, score.size()[0], num_negs + 1)]
        neg_score = unstacked_scores[torch.arange(unstacked_scores.size()[0]),1:,:]
        neg_score = (nn.functional.softmax(neg_score * alpha, dim=1).detach()
                    * nn.functional.logsigmoid(-neg_score)).squeeze().sum(dim=1)
        pos_score = nn.functional.logsigmoid(pos_score).squeeze()
        positive_sample_loss = - pos_score.mean()
        negative_sample_loss = - neg_score.mean()
        
        loss = (positive_sample_loss + negative_sample_loss) / 2
        return loss

    def compute_rp_loss(self, logits, targets, num_negatives: int):
        """
        CrossEntropy loss:
        - if negatives > 0: take only the positive indices and average
        - else: mean over the whole batch
        """
        loss_vec = self.rp_loss_fn(logits, targets)
        if num_negatives > 0:
            assert loss_vec.size(0) % (num_negatives + 1) == 0, \
                "Incompatible loss size and negative sample sizes."
            pos_idx = torch.arange(0, loss_vec.size(0), num_negatives + 1, device=loss_vec.device)
            return loss_vec[pos_idx].mean()
        return loss_vec.mean()

    def compute_lp_loss(self, *args, **kwargs):
        return self.lp_loss_fn(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Forward pass
        if self.train_sub_batch_size is None:
            logits_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, logits_lp = self.sub_batch(batch=batch, sub_batch_size=self.train_sub_batch_size)

        targets = batch["target"] - 2  # no PAD and MSK in the model output

        # Compute losses
        rp_loss = self.compute_rp_loss(logits_rp, targets, self.train_num_negatives)

        if "lp_labels" in batch:
            assert(self.train_num_negatives == 0), "train_num_negatives should be 0 when using provided lp_labels and lp_weights!"
            lp_loss = self.calculate_lp_bce(logits_lp, labels=batch["lp_labels"], sample_weights=batch["lp_weights"], num_negatives=None)
            loss = (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss)
            self.log("train_loss_rp", rp_loss)
            self.log("train_loss_lp", lp_loss, prog_bar=True)
            self.log("train_loss", loss, prog_bar=self.loss_weight not in [0.0, 1.0])
            if torch.isnan(loss):
                logger.warning("Training stopped due to NaN loss.")
                self.trainer.should_stop = True
                self.trainer.limit_val_batches = 0
            return loss
        elif self.train_num_negatives > 0:
            lp_loss = self.compute_lp_loss(logits_lp, self.train_num_negatives)
            loss = (self.loss_weight * lp_loss) +  ((1.0 - self.loss_weight) * rp_loss)
            self.log("train_loss_rp", rp_loss)
            self.log("train_loss_lp", lp_loss)
            self.log("train_loss", loss, prog_bar=True)
            if torch.isnan(loss):
                logger.warning("Training stopped due to NaN loss.")
                self.trainer.should_stop = True
                self.trainer.limit_val_batches = 0
            return loss
        else:
            self.log("train_loss_rp", rp_loss)
            self.log("train_loss", rp_loss, prog_bar=True)
            return rp_loss
        # make_dot(loss, params=dict(self.model.named_parameters())).render(
        #     "attachedavg", format="png")

    def calculate_and_log_val_relation_metrics(self, triples, logits):
        self.val_relationMRR(triples=triples,
                             scores=logits)
        self.val_relationHitsAt1(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt3(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt5(triples=triples,
                                 scores=logits)
        self.val_relationHitsAt10(triples=triples,
                                  scores=logits)
        self.log("valid_mrr", self.val_relationMRR, on_step=False,
                 on_epoch=True)
        self.log("valid_hits1", self.val_relationHitsAt1, on_step=False,
                 on_epoch=True)
        self.log("valid_hits3", self.val_relationHitsAt3, on_step=False,
                 on_epoch=True)
        self.log("valid_hits5", self.val_relationHitsAt5, on_step=False,
                 on_epoch=True)
        self.log("valid_hits10", self.val_relationHitsAt10, on_step=False,
                 on_epoch=True)

    def calculate_and_log_test_relation_metrics(self, triples, logits):
        self.test_relationMRR(triples=triples,
                              scores=logits)
        self.test_relationHitsAt1(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt3(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt5(triples=triples,
                                  scores=logits)
        self.test_relationHitsAt10(triples=triples,
                                   scores=logits)
        self.log("test_mrr", self.test_relationMRR, on_step=False,
                 on_epoch=True)
        self.log("test_hits1", self.test_relationHitsAt1, on_step=False,
                 on_epoch=True)
        self.log("test_hits3", self.test_relationHitsAt3, on_step=False,
                 on_epoch=True)
        self.log("test_hits5", self.test_relationHitsAt5, on_step=False,
                 on_epoch=True)
        self.log("test_hits10", self.test_relationHitsAt10, on_step=False,
                 on_epoch=True)

    def calculate_and_log_val_links_metrics(self, rp_triples, logits_rp):
        if self.val_num_negatives == 0:
            return
        self.val_linkMRR(triples=rp_triples, scores=logits_rp,
                         num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt1(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt3(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt5(triples=rp_triples, scores=logits_rp,
                             num_entities_per_sample=self.val_num_negatives)
        self.val_linkHitsAt10(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.val_num_negatives)
        self.log("valid_link_mrr", self.val_linkMRR, on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits1", self.val_linkHitsAt1,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits3", self.val_linkHitsAt3,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits5", self.val_linkHitsAt5,
                 on_step=False,
                 on_epoch=True)
        self.log("valid_link_hits10", self.val_linkHitsAt10,
                 on_step=False,
                 on_epoch=True)

    def calculate_and_log_test_links_metrics(self, rp_triples, logits_rp):
        if self.test_num_negatives == 0:
            return
        self.test_linkMRR(triples=rp_triples, scores=logits_rp,
                          num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt1(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt3(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt5(triples=rp_triples, scores=logits_rp,
                              num_entities_per_sample=self.test_num_negatives)
        self.test_linkHitsAt10(triples=rp_triples, scores=logits_rp,
                               num_entities_per_sample=self.test_num_negatives)
        self.log("test_link_mrr", self.test_linkMRR, on_step=False,
                 on_epoch=True)
        self.log("test_link_hits1", self.test_linkHitsAt1,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits3", self.test_linkHitsAt3,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits5", self.test_linkHitsAt5,
                 on_step=False,
                 on_epoch=True)
        self.log("test_link_hits10", self.test_linkHitsAt10,
                 on_step=False,
                 on_epoch=True)

    def calculate_and_log_candidate_metrics(self, scores: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor, split: str = "valid"):
        """
        Update torchmetrics dict (epoch-level) and log:
          - {split}_link_mrr
          - {split}_link_hits@{k}
          - {split}_link_recall@{k}_perGroup
        """
        # select the metrics dict
        if split == "valid":
            metrics = self.cand_metrics_val
        elif split == "test":
            metrics = self.cand_metrics_test
        else:
            raise ValueError("split must be 'val' or 'test'")
        
        # Ensure tensors on module device
        scores = scores.detach().squeeze().to(self.device)
        labels = labels.detach().to(self.device)
        group_ids = group_ids.detach().to(self.device)

        # update and log all metrics
        for key, metric in metrics.items():
            metric.update(scores=scores, labels=labels, group_ids=group_ids)
            self.log(f"{split}_link_{key}", metric, on_step=False, on_epoch=True, prog_bar=("mrr" in key))

    def sub_batch(self, batch, sub_batch_size):
        """
        A function that sub batches the evaluation so that it fits into GPU
        memory

        Parameters
        ----------
        batch : The batch of network inputs

        """
        # get the sub batch size (in number of paths)
        sub_batch_size = sub_batch_size
        # get the device to send the tensors after accumulation of sub batches
        model_device = self.device
        # get the number of output neurons of the prediction head
        rp_out_size = self.model.rel_embeddings.weight.size()[0] - 2
        # get the size of the batch (in triples)
        batch_size = batch['target'].size()
        # initialize tensors to aggregate the sub_batches
        # these are initialized using the number of triples as we cannot
        # separate the paths belonging to each triple (would mess up the
        # attention and aggregation)
        logits_rp = torch.zeros((batch_size[0], rp_out_size), dtype=torch.float32)
        logits_lp = torch.zeros((batch_size[0]), dtype=torch.float32)
        # get the paths per triple
        ppt = batch["ppt"]
        # if the sub_batch size is smaller than the max number of paths per
        # triple, increase it to the max number of paths and print a warning
        # if sub_batch_size < ppt.max():
        #     warnings.warn("Sub-batch size cannot be less than the maximum "
        #                   "paths per "
        #                   "triple, setting sub-batch to: " + str(ppt.max()))
        #     sub_batch_size = ppt.max()
        # initialize indexing variable to get the batches
        # the cumulative sum is used to know how many paths are included when
        # grabbing more than one triple in the subbatch
        ppt_cum_sum = torch.cumsum(ppt, dim=0)
        remaining = ppt_cum_sum
        current_idx_cum_sum = 0
        initial = 0
        initial_triple = 0
        # as long as the remaining paths (last element) is not zero
        while remaining[-1] != 0:
            # start getting triples as long as the number of paths is smaller
            # than the subbatch size
            for i in range(remaining.size()[0]):
                if remaining[i] > sub_batch_size:
                    break
                else:
                    current_idx_cum_sum = i
            # once we found the last index of the triples that can be
            # included in the sub batch create the offsets, so we can slice
            # the batch
            path_offset = remaining[current_idx_cum_sum]
            remaining = remaining - path_offset
            triple_offset = current_idx_cum_sum + 1  # necessary as upper end
            # in slicing is exclusive
            # create the sub batch dictionary
            sub_batch = {
                "id": batch["id"][initial_triple:triple_offset],
                "ppt": batch["ppt"][initial_triple:triple_offset],
                "net_input": {
                    "ent_paths": batch["net_input"]["ent_paths"][initial: initial + path_offset, :],
                    "rel_paths": batch["net_input"]["rel_paths"][initial: initial + path_offset, :],
                    "ent_lengths": batch["net_input"]["ent_lengths"][initial: initial + path_offset],
                    "head_idxs": batch["net_input"]["head_idxs"][initial: initial + path_offset],
                    "tail_idxs": batch["net_input"]["tail_idxs"][initial: initial + path_offset],
                    "path_origins": batch["net_input"]["path_origins"][initial: initial + path_offset],
                    "pos": batch["net_input"]["pos"][initial: initial + path_offset, :],
                },
                "target": batch["target"][initial_triple: triple_offset],
                "ori_triple": batch["ori_triple"][initial_triple: triple_offset, :],
            }

            # get the model outputs and losses for the sub batch
            sb_logits_rp, sb_logits_lp = self.model_forward(batch=sub_batch)
            # aggregate the results in the aggregation tensors
            logits_rp[initial_triple: triple_offset, :] = sb_logits_rp
            logits_lp[initial_triple: triple_offset] = sb_logits_lp.squeeze()

            # increase the starting indices to slice the next sub_batch
            initial = initial + path_offset
            initial_triple = triple_offset
        return (logits_rp.to(torch.device(model_device)),
                logits_lp.to(torch.device(model_device)))

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.val_sub_batch_size is None:
            logits_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, logits_lp = self.sub_batch(batch=batch, sub_batch_size=self.val_sub_batch_size)

        targets: torch.Tensor = batch["target"] - 2
        triples: torch.Tensor = batch['ori_triple']

        # RP loss (always)
        rp_loss = self.compute_rp_loss(logits_rp, targets, self.val_num_negatives)
        self.log("valid_rp_loss", rp_loss, prog_bar=True)

        # Candidate mode if labels present
        if "lp_labels" in batch:
            lp_labels = batch["lp_labels"]
            lp_weights = batch.get("lp_weights")
            lp_groups = batch.get("lp_group_ids")

            lp_loss = self.calculate_lp_bce(logits_lp, labels=lp_labels, sample_weights=lp_weights, num_negatives=None)
            self.log("valid_lp_loss", lp_loss, prog_bar=True)
            self.log("valid_total_loss", (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss),
                     prog_bar=self.loss_weight not in [0.0, 1.0])

            # Accumulate everything into unified _val_acc
            if not hasattr(self, "_val_acc"):
                assert batch_idx == 0, "Accumulation not cleared from previous epoch."
                self._val_acc = {
                    "triples_rp": [], "logits_rp": [],
                    "logits_lp": [], "lp_labels": [], "lp_groups": [],
                }
            # Only keep positives for relation metrics to save RAM
            pos_mask = (lp_labels > 0.5)
            self._val_acc["triples_rp"].append(triples[pos_mask].detach().cpu())
            self._val_acc["logits_rp"].append(logits_rp[pos_mask].detach().cpu())
            # Keep all candidates for link metrics
            self._val_acc["logits_lp"].append(logits_lp.detach().cpu())
            self._val_acc["lp_labels"].append(lp_labels.detach().cpu())
            self._val_acc["lp_groups"].append(lp_groups.detach().cpu())
            return

        # Legacy LP loss/accumulation
        if self.val_num_negatives > 0:
            lp_loss = self.compute_lp_loss(logits_lp, self.val_num_negatives)
            self.log("valid_lp_loss", lp_loss, prog_bar=True)
            self.log("valid_total_loss", (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss), prog_bar=True)

            if not hasattr(self, "_val_acc"):
                assert(batch_idx == 0), "Accumulation not cleared from previous epoch."
                self._val_acc = {"triples_rp": [], "triples_lp": [], "logits_rp": [], "logits_lp": []}

            # select only positives for relation metrics
            assert logits_rp.size(0) % (self.val_num_negatives + 1) == 0, "Incompatible batch and negatives."
            pos_idx = torch.arange(0, logits_rp.size(0), self.val_num_negatives + 1)
            logits_rp_only = logits_rp[pos_idx]
            triples_rp_only = triples[pos_idx]
            self._val_acc["triples_rp"].append(triples_rp_only.detach().cpu())
            self._val_acc["triples_lp"].append(triples.detach().cpu())
            self._val_acc["logits_rp"].append(logits_rp_only.detach().cpu())
            self._val_acc["logits_lp"].append(logits_lp.detach().cpu())
        else:
            self.log("valid_total_loss", rp_loss, prog_bar=True)
            if not hasattr(self, "_val_acc"):
                assert(batch_idx == 0), "Accumulation not cleared from previous epoch."
                self._val_acc = {"triples_rp": [], "logits_rp": []}
            self._val_acc["triples_rp"].append(triples.detach().cpu())
            self._val_acc["logits_rp"].append(logits_rp.detach().cpu())

    def validation_epoch_end(self, outputs):
        if not hasattr(self, "_val_acc"):
            logger.warning("validation_epoch_end() called without any accumulated validation data!")
            return
        else:
            if "lp_labels" in self._val_acc and "logits_lp" in self._val_acc and "lp_groups" in self._val_acc:
                # Candidate mode metrics
                triples_rp = torch.cat(self._val_acc["triples_rp"], dim=0)
                logits_rp = torch.cat(self._val_acc["logits_rp"], dim=0)
                lp_labels = torch.cat(self._val_acc["lp_labels"], dim=0)
                logits_lp = torch.cat(self._val_acc["logits_lp"], dim=0)
                lp_groups = torch.cat(self._val_acc["lp_groups"], dim=0)

                # make sure sizes are compatible and everything went well in val step
                assert(logits_rp.size(0) == triples_rp.size(0)), 'Incompatible batch and logits sizes, perhaps something went wrong while generating them in validation_step().'
                assert(logits_lp.size(0) == lp_labels.size(0) == lp_groups.size(0)), 'Incompatible candidate scores, labels or groups sizes, perhaps something went wrong while generating them in validation_step().'

                # Relation metrics
                self.calculate_and_log_val_relation_metrics(triples_rp, logits_rp)

                # Candidate metrics
                self.calculate_and_log_candidate_metrics(logits_lp, lp_labels, lp_groups, split="valid")
            else:
                # Legacy RP/LP metrics
                triples_rp = self._val_acc["triples_rp"]
                triples_lp = self._val_acc["triples_lp"]
                logits_rp = self._val_acc["logits_rp"]
                logits_lp = self._val_acc["logits_lp"]

                if len(triples_rp) >= 0 and len(logits_rp) >= 0:
                    triples_rp = torch.cat(triples_rp, dim=0)
                    logits_rp = torch.cat(logits_rp, dim=0)
                    assert(logits_rp.size(0) == triples_rp.size(0), 'Incompatible batch and logits sizes, perhaps something went wrong while generating them in validation_step().')
                    self.calculate_and_log_val_relation_metrics(triples_rp, logits_rp)
                if len(triples_lp) > 0 and len(logits_lp) > 0:
                    triples_lp = torch.cat(triples_lp, dim=0)
                    logits_lp = torch.cat(logits_lp, dim=0)
                    assert(logits_lp.size(0) == triples_lp.size(0), 'Incompatible batch and logits sizes, perhaps something went wrong while generating them in validation_step().')
                    self.calculate_and_log_val_links_metrics(triples_lp, logits_lp)
            del self._val_acc  # delete reference for next validation epoch

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.test_sub_batch_size is None:
            logits_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, logits_lp = self.sub_batch(batch=batch, sub_batch_size=self.test_sub_batch_size)

        targets = batch["target"] - 2
        triples = batch['ori_triple']

        rp_loss = self.compute_rp_loss(logits_rp, targets, self.test_num_negatives)
        self.log("test_rp_loss", rp_loss, prog_bar=True)

        # Candidate mode per-batch metrics
        if "lp_labels" in batch:
            lp_labels = batch["lp_labels"]
            lp_weights = batch.get("lp_weights")
            lp_groups = batch.get("lp_group_ids")

            lp_loss = self.calculate_lp_bce(logits_lp, labels=lp_labels, sample_weights=lp_weights, num_negatives=None)
            self.log("test_lp_loss", lp_loss, prog_bar=True)
            self.log("test_total_loss", (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss),
                     prog_bar=self.loss_weight not in [0.0, 1.0])

            # Accumulate for epoch-end heavy metrics
            if not hasattr(self, "_test_acc"):
                assert batch_idx == 0, "Test accumulation not cleared from previous epoch."
                self._test_acc = {"triples_rp": [], "logits_rp": [], "logits_lp": [], "lp_labels": [], "lp_groups": []}
            
            # Only keep positives for relation metrics to save RAM
            pos_mask = (lp_labels > 0.5)
            self._test_acc["triples_rp"].append(triples[pos_mask].detach().cpu())
            self._test_acc["logits_rp"].append(logits_rp[pos_mask].detach().cpu())
            # Keep all candidates for link metrics
            self._test_acc["logits_lp"].append(logits_lp.detach().cpu())
            self._test_acc["lp_labels"].append(lp_labels.detach().cpu())
            self._test_acc["lp_groups"].append(lp_groups.detach().cpu())
            return
        
        # Legacy test path
        elif self.test_num_negatives > 0:
            lp_loss = self.compute_lp_loss(logits_lp, self.test_num_negatives)
            self.log("test_lp_loss", lp_loss, prog_bar=True)
            self.log("test_total_loss", (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss), prog_bar=True)

            assert logits_rp.size(0) % (self.test_num_negatives + 1) == 0, "Incompatible batch and negatives."
            pos_idx = torch.arange(0, logits_rp.size(0), self.test_num_negatives + 1)
            logits_rp_only = logits_rp[pos_idx]
            triples_rp_only = triples[pos_idx]

            self.calculate_and_log_test_relation_metrics(triples_rp_only, logits_rp_only)
            self.calculate_and_log_test_links_metrics(triples, logits_lp)
        else:
            self.log("test_total_loss", rp_loss, prog_bar=True)
            self.calculate_and_log_test_relation_metrics(triples, logits_rp)

    def on_test_epoch_end(self):
        if not hasattr(self, "_test_acc"):
            return
        triples_rp = torch.cat(self._test_acc["triples_rp"], dim=0)
        logits_rp = torch.cat(self._test_acc["logits_rp"], dim=0)
        logits_lp = torch.cat(self._test_acc["logits_lp"], dim=0)
        lp_labels = torch.cat(self._test_acc["lp_labels"], dim=0)
        lp_groups = torch.cat(self._test_acc["lp_groups"], dim=0)

        # Relation metrics (epoch-level)
        self.calculate_and_log_test_relation_metrics(triples_rp, logits_rp)

        # Candidate metrics (epoch-level)
        self.calculate_and_log_candidate_metrics(logits_lp, lp_labels, lp_groups, split="test")

        del self._test_acc  # delete reference for next test epoch

    def configure_optimizers(self):
        """
        Called by PL-Lightning to initialise and setup optimisers and learning
        rate schedulers. Currently supports basic optims and functionalities.

        Note: most of this function can be implemented in a util.
        """
        optim_hps = dict(params=self.model.parameters(),
                         lr=self.lr,  # initial if scheduler
                         weight_decay=self.weight_decay)
        optimiser = None  # one of the supported optimisers below
        if self.optimiser == "adam":
            optimiser = torch.optim.Adam(
                **optim_hps)
        elif self.optimiser == "sgd":
            optimiser = torch.optim.SGD(
                **optim_hps,
                momentum=self.momentum)
        elif self.optimiser == "rms":
            optimiser = torch.optim.RMSprop(
                **optim_hps,
                momentum=self.momentum)
        else:  # not supported or implemented
            raise ValueError(f"Not a valid optimiser: {self.optimiser}")

        opt_dict = {"optimizer": optimiser}
        if self.scheduler != "none":
            raise NotImplementedError()
        return opt_dict  # makes distinction among optimisers and schedulers
    
    def model_forward(self, batch):
        return self.pathe_forward_step_triples(batch=batch)

    def pathe_forward_step_triples(self, batch, **kwargs):
        """
        PathE forward function for multi-paths relation prediction.
        """
        # batch = debug_single_path(batch, trim_paths=False, override_pos=True)
        ent_paths = batch["net_input"]["ent_paths"]
        rel_paths = batch["net_input"]["rel_paths"]
        head_idxs = batch["net_input"]["head_idxs"]
        tail_idxs = batch["net_input"]["tail_idxs"]
        entity_origin = batch["net_input"]["path_origins"]
        pos = batch["net_input"]["pos"]
        targets = batch["target"] - 2  # no PAD and MSK in the model output
        ppt = batch["ppt"]

        # FIXME this is going to change with the actual head-tail idxs from dataset
        head_idxs = head_idxs * 2
        tail_idxs = tail_idxs * 2

        logits_rp, logits_lp = self.model(
            ent_paths=ent_paths,
            rel_paths=rel_paths,
            head_idxs=head_idxs,
            tail_idxs=tail_idxs,
            ppt=ppt, 
            pos=pos,
            entity_origin=entity_origin,
            targets=targets)
        return logits_rp, logits_lp


class PathEModelWrapperTuples(PathEModelWrapperTriples):
    """
    PL Wrapper for training PathE with tuples.

    """

    def __init__(self, 
                 pathe_model: PathEModelTuples, filtration_dict, num_negatives=0,
                 optimiser="adam", scheduler="none", lr=1e-3, momentum=0,
                 weight_decay=0, class_weights=None, label_smoothing=0.0,
                 train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                 val_num_negatives=0, full_test=False, max_ppt=None, accumulate_gradient=1.0, 
                 margin=10, nssa_alpha=1, lp_loss_fn="nssa", link_head_detached: bool = True, use_manual_optimization: bool = False,
                 loss_weight: float = 0.5, 
                train_relation_maps: RelationMaps = None,
                val_relation_maps: RelationMaps = None,
                test_relation_maps: RelationMaps = None,
                 **hparams):
        super().__init__(pathe_model, filtration_dict, num_negatives,
                         optimiser, scheduler, lr, momentum, weight_decay,
                         class_weights, label_smoothing,
                         train_sub_batch=train_sub_batch,
                         val_sub_batch=val_sub_batch,
                         test_sub_batch=test_sub_batch,
                         val_num_negatives=val_num_negatives,
                         full_test=full_test, max_ppt=max_ppt,
                         margin=margin, nssa_alpha=nssa_alpha,
                         lp_loss_fn=lp_loss_fn, loss_weight=loss_weight)

        self.accumulate_gradient = accumulate_gradient
        self.use_manual_optimization = use_manual_optimization
        self.link_head_detached = link_head_detached
        self.train_relation_maps = train_relation_maps
        self.val_relation_maps = val_relation_maps
        self.test_relation_maps = test_relation_maps

        # Set automatic optimization based on parameter
        self.automatic_optimization = not self.use_manual_optimization

        # unnecessary as done in super but for clarity
        self.model = pathe_model

        # define loss functions which have not been defined in super().__init__()
        self.rp_loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            reduction='none',
            label_smoothing=label_smoothing)

        # override the metrics set in the parent class to now evaluate tuples
        # [List of metrics we will be watching on validation and test sets]
        self.val_relationMRR = RelationMRRTuples(filtration_dict)
        self.val_relationHitsAt1 = RelationHitsAtKTuples(filtration_dict, k=1)
        self.val_relationHitsAt3 = RelationHitsAtKTuples(filtration_dict, k=3)
        self.val_relationHitsAt5 = RelationHitsAtKTuples(filtration_dict, k=5)
        self.val_relationHitsAt10 = RelationHitsAtKTuples(filtration_dict, k=10)
        self.test_relationMRR = RelationMRRTuples(filtration_dict)
        self.test_relationHitsAt1 = RelationHitsAtKTuples(filtration_dict, k=1)
        self.test_relationHitsAt3 = RelationHitsAtKTuples(filtration_dict, k=3)
        self.test_relationHitsAt5 = RelationHitsAtKTuples(filtration_dict, k=5)
        self.test_relationHitsAt10 = RelationHitsAtKTuples(filtration_dict, k=10)

        # watch link prediction metrics
        if self.val_num_negatives > 0:  
            # raise NotImplementedError("check whether tuples versions are needed to be used. ")
            self.val_linkMRR = EntityMRRTriples()
            self.val_linkHitsAt1 = EntityHitsAtKTriples(k=1)
            self.val_linkHitsAt3 = EntityHitsAtKTriples(k=3)
            self.val_linkHitsAt5 = EntityHitsAtKTriples(k=5)
            self.val_linkHitsAt10 = EntityHitsAtKTriples(k=10)
            self.test_linkMRR = EntityMRRTriples()
            self.test_linkHitsAt1 = EntityHitsAtKTriples(k=1)
            self.test_linkHitsAt3 = EntityHitsAtKTriples(k=3)
            self.test_linkHitsAt5 = EntityHitsAtKTriples(k=5)
            self.test_linkHitsAt10 = EntityHitsAtKTriples(k=10)

    def configure_optimizers(self):
        if self.use_manual_optimization:
            # perhaps consider adding other optimizers as in triples version
            assert(self.optimiser == "adam"), "Currently only Adam is supported for manual optimization."
            # Return separate optimizers for relation and link prediction
            relation_params = []
            link_params = []
            shared_params = []
            
            # Separate parameters based on their role
            for name, param in self.model.named_parameters():
                if 'relpredict_head_avg' in name:
                    relation_params.append(param)
                elif 'link_predict_head' in name:
                    link_params.append(param)
                else:
                    shared_params.append(param)
            
            relation_optimizer = torch.optim.Adam(
                shared_params + relation_params, 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
            
            link_optimizer = torch.optim.Adam(
                shared_params + link_params if not self.link_head_detached else link_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            
            return [relation_optimizer, link_optimizer]
        else:
            return super().configure_optimizers()

    def compute_rp_loss(self, logits, targets, num_negatives):
        loss_rp = self.rp_loss_fn(logits, targets)
        assert(loss_rp.size(0) % (num_negatives + 1) == 0), "Incompatible loss size and negative sample sizes probably something when wrong with the batch size when generating negatives!"
        loss_rp = torch.mean(loss_rp[torch.arange(0, loss_rp.size()[0], num_negatives + 1)])
        return loss_rp

    def compute_lp_loss(self, logits, num_negatives):
        loss_lp = self.lp_loss_fn(logits, num_negatives)
        return loss_lp

    def training_step(self, batch, batch_idx):
        if not self.use_manual_optimization:
            return super().training_step(batch, batch_idx)
        
        # Manual optimization mode
        relation_opt, link_opt = self.optimizers()
        
        # Forward pass
        logits_rp, logits_lp = self.model_forward(batch)

        # # generate triple candidates
        # tuples = batch['ori_triple']
        # triple_candidates, _val = self.build_triple_candidates(tuples, self.train_relation_maps, logits_rp, k=100)
        # print(f"Generated {triple_candidates.size(0)} triple candidates for {tuples.size(0)} input tuples: {triple_candidates[:5,:]} ..., {_val[:5]} ...")

        # Calculate separate losses
        targets = batch["target"] - 2
        rp_loss = self.compute_rp_loss(logits_rp, targets, self.train_num_negatives)
        lp_loss = self.compute_lp_loss(logits_lp, self.train_num_negatives)
        if torch.isnan(rp_loss) or torch.isnan(lp_loss):
            logger.warning(f"Training stopped due to NaN loss: rp_loss: {rp_loss}, lp_loss: {lp_loss}.")
            self.trainer.should_stop = True
            self.trainer.limit_val_batches = 0

        # Scale losses for accumulation
        rp_loss = rp_loss / self.accumulate_gradient
        lp_loss = lp_loss / self.accumulate_gradient

        # Accumulate gradients
        # Backward passes:
        # 1) Propagate relation loss through the entire model so shared
        #    parameters and the relation head receive gradients.
        #    By using retain_graph=True here we keep the graph for a second backward.
        self.manual_backward(rp_loss, retain_graph=True)
        # 2) Propagate link loss only through the link head.
        #    When link_head_detached=True the link optimizer was created with
        #    link-head-only params; in that case this backward should therefore only affect
        #    link-head parameters (and not embeddings) as intended.
        if lp_loss.item() > 0:
            self.manual_backward(lp_loss)
        
        # Step and zero grads when enough batches have been accumulated
        if (batch_idx + 1) % self.accumulate_gradient == 0:
            # Optimize relation prediction
            self.clip_gradients(relation_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            relation_opt.step()
            relation_opt.zero_grad()

            # Optimize link prediction  
            if lp_loss.item() > 0:
                self.clip_gradients(link_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                link_opt.step()
                link_opt.zero_grad()
        
        # Logging
        self.log("train_rp_loss", rp_loss, prog_bar=True)
        self.log("train_lp_loss", lp_loss, prog_bar=(self.train_num_negatives > 0))
        self.log("train_total_loss", rp_loss + lp_loss)
        
        return {"rp_loss": rp_loss, "lp_loss": lp_loss}
    
    def validation_step(self, batch, batch_idx):
        if not self.use_manual_optimization:
            return super().validation_step(batch, batch_idx)

        # Forward pass
        logits_rp, logits_lp = self.model_forward(batch)
        targets = batch["target"] - 2
        tuples = batch['ori_triple']

        # Compute losses
        rp_loss = self.compute_rp_loss(logits_rp, targets, self.val_num_negatives)
        lp_loss = self.compute_lp_loss(logits_lp, self.val_num_negatives)

        # Logging losses
        self.log("valid_rp_loss", rp_loss, prog_bar=True)
        self.log("valid_lp_loss", lp_loss, prog_bar=(self.val_num_negatives > 0))
        self.log("valid_total_loss", rp_loss + lp_loss)

        # Metrics for tuples (relation prediction)
        assert(tuples.size()[0] == logits_rp.size()[0])
        logits_rp_only_positives = logits_rp[torch.arange(0, logits_rp.size()[0], self.val_num_negatives + 1)]
        tuples_only_positives = tuples[torch.arange(0, tuples.size()[0], self.val_num_negatives + 1)]

        # Accumulate for epoch-level metric computation (avoid per-batch heavy metrics)
        if not hasattr(self, "_val_acc"):
            assert(batch_idx == 0), "Somehow accumulating validation data was not deleted in validation_epoch_end() after last epoch. Risking incorrect metrics!"
            self._val_acc = {"tuples_rp": [], "tuples_lp": [], "logits_rp": [], "logits_lp": []}
        else:
            assert(batch_idx > 0), "Somehow accumulating validation data was not initialized in previous validation_steps. Risking incorrect metrics!"
        self._val_acc["tuples_rp"].append(tuples_only_positives.detach().cpu())
        self._val_acc["tuples_lp"].append(tuples.detach().cpu())
        self._val_acc["logits_rp"].append(logits_rp_only_positives.detach().cpu())
        self._val_acc["logits_lp"].append(logits_lp.detach().cpu())

        return {"rp_loss": rp_loss, "lp_loss": lp_loss}

    def validation_epoch_end(self, outputs):
        if not hasattr(self, "_val_acc"):
            logger.warning("validation_epoch_end() called without any accumulated validation data!")
            return
        # Rename accumulated validation data for usage in triples' metric calculation function
        if "tuples_rp" in self._val_acc:
            self._val_acc['triples_rp'] = self._val_acc.pop('tuples_rp')
        if "tuples_lp" in self._val_acc:
            self._val_acc['triples_lp'] = self._val_acc.pop('tuples_lp')
        return super().validation_epoch_end(outputs)

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        logits_rp, logits_lp = self.pathe_forward_step_tuples(batch)
        tuples = batch['ori_triple']  # (num_samples, 2) -> (h, r)
        return {
            "tuples": tuples.detach().cpu(),
            "logits_rp": logits_rp.detach().cpu(),
            "logits_lp": logits_lp.detach().cpu()
        }
    
    
    # quite similar to validation_step but with different logging names and metric calculation functions
    def test_step(self, batch, batch_idx):
        if not self.use_manual_optimization:
            return super().test_step(batch, batch_idx)

        # Forward pass
        logits_rp, logits_lp = self.model_forward(batch)
        targets = batch["target"] - 2
        tuples = batch['ori_triple']

        # Compute losses
        rp_loss = self.compute_rp_loss(logits_rp, targets, self.test_num_negatives)
        lp_loss = self.compute_lp_loss(logits_lp, self.test_num_negatives)

        # Logging losses
        self.log("test_rp_loss", rp_loss, prog_bar=True)
        self.log("test_lp_loss", lp_loss, prog_bar=True)
        self.log("test_total_loss", rp_loss + lp_loss)

        # Metrics for tuples (relation prediction)
        assert(tuples.size()[0] == logits_rp.size()[0])
        logits_rp_only_positives = logits_rp[torch.arange(0, logits_rp.size()[0], self.test_num_negatives + 1)]
        tuples_only_positives = tuples[torch.arange(0, tuples.size()[0], self.test_num_negatives + 1)]
        self.calculate_and_log_test_relation_metrics(tuples_only_positives, logits_rp_only_positives)
        self.calculate_and_log_test_links_metrics(tuples, logits_lp)
    
    def model_forward(self, batch):
        return self.pathe_forward_step_tuples(batch)

    def pathe_forward_step_tuples(self, batch, **kwargs):
        """
        PathE forward function for multi-paths relation prediction.
        """
        # batch = debug_single_path(batch, trim_paths=False, override_pos=True)
        ent_paths, rel_paths = batch["net_input"]["ent_paths"], batch["net_input"]["rel_paths"]
        head_idxs = batch["net_input"]["head_idxs"]
        entity_origin = batch["net_input"]["path_origins"]
        pos = batch["net_input"]["pos"]
        targets = batch["target"] - 2  # no PAD and MSK in the model output
        # FIXME this is going to change with the actual head-tail idxs from dataset
        head_idxs = head_idxs * 2
        ppt = batch["ppt"]

        logits_rp, logits_lp = self.model(
            ent_paths=ent_paths,
            rel_paths=rel_paths,
            head_idxs=head_idxs,
            ppt=ppt,
            pos=pos,
            entity_origin=entity_origin,
            targets=targets,
            detach_link_head=self.link_head_detached
        )
        # loss_rp = self.criterion(logits_rp, targets)
        return logits_rp, logits_lp
    
    def build_triple_candidates_adaptive(
        self,
        tuples: torch.Tensor,
        relation_maps: RelationMaps,
        logits_rp: torch.Tensor,
        p: float = None,          # keep candidates with P >= p (global threshold)
        q: float = None,          # use as fraction-cap: cap = ceil((1-q) * |E|*|R|*|E|)
        temperature: float = 1.0, # temperature for softmax calibration
        alpha: float = 0.5,       # weight for head vs tail log-probs
        cap_candidates: int = None # final cap after thresholding only keep top-k candidates
    ):
        """
        Build candidate (head, relation, tail) triples and their joint probabilities.

        Selection logic (global, not per-entity):
          1) Compute an effective cap from q: cap_q = ceil((1-q) * E*R*E). If provided,
             cap_candidates = min(cap_candidates or inf, cap_q).
          2) If p is provided, keep all with P >= p (global threshold).
          3) If more than cap_candidates remain, keep the top cap_candidates globally.
          4) If cap_candidates is None and p is None, keep all.

        Args:
            tuples: (num_samples, 2) with entity in col 0.
            relation_maps: mapping for original/inverse relations.
            logits_rp: per-sample logits (num_samples, total_relations).
            p: float or None, global probability threshold.
            q: float in [0,1), quantile threshold: keeps top (1-q) fraction by joint probability,
               implemented via top-k selection (not by explicit quantile computation).
            temperature: softmax temperature (>0).
            alpha: weight in [0,1] for head vs tail contribution.
            cap_candidates: optional int final cap.

        Returns:
            candidates: (N,3) tensor of (head_id, relation_id, tail_id)
            scores: (N,) joint probabilities
        """
        assert logits_rp is not None, "logits_rp required."
        assert temperature > 0, "temperature must be > 0"
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0,1]"

        # 1. Collect unique head entities (local indexing)
        entities, inverse_entity_indices = tuples[:, 0].unique(return_inverse=True, sorted=False)
        E = entities.size(0)

        # 2. Aggregate logits per local head index
        logits_rp_grouped = torch_scatter.scatter_mean(logits_rp, inverse_entity_indices, dim=0)
        device = logits_rp_grouped.device

        # 3. Resolve original & inverse relation ids
        original_relations = torch.tensor(list(relation_maps.original_relations), device=device, dtype=torch.long)
        inverse_relations = torch.tensor(list(relation_maps.inverse_relations), device=device, dtype=torch.long)
        assert original_relations.numel() == inverse_relations.numel(), "Mismatch originals/inverses."
        R = original_relations.size(0)

        # 4. Slice logits for original and inverse relation columns
        head_logits_subset = logits_rp_grouped[:, original_relations]   # (E, R)
        tail_logits_subset = logits_rp_grouped[:, inverse_relations]    # (E, R)

        # 5. Calibrated log-probabilities (avoid tiny exp, use log_softmax)
        log_p_head = torch.log_softmax(head_logits_subset / temperature, dim=1).unsqueeze(2)                 # (E, R, 1)
        log_p_tail = torch.log_softmax(tail_logits_subset / temperature, dim=1).transpose(0, 1).unsqueeze(0) # (1, R, E)

        # 6. Joint log-prob with weighting
        log_joint = alpha * log_p_head + (1.0 - alpha) * log_p_tail  # (E, R, E)

        # Flatten once for global selection
        log_joint_flat = log_joint.reshape(-1)  # (E*R*E,)
        total = log_joint_flat.numel()

        # Derive effective cap from q first
        effective_cap = cap_candidates
        if q is not None:
            q = float(q)
            assert 0.0 <= q < 1.0, "q must be in [0,1)"
            cap_q = max(1, int(math.ceil((1.0 - q) * total)))
            effective_cap = cap_q if effective_cap is None else min(effective_cap, cap_q)

        # Step A: optional probability threshold
        if p is not None:
            log_p_thr = torch.tensor(float(p), device=device).clamp_min(1e-45).log()
            mask = log_joint_flat >= log_p_thr
            flat_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if flat_indices.numel() == 0:
                # fallback to best single candidate
                _, top_idx = torch.topk(log_joint_flat, k=1, largest=True)
                flat_indices = top_idx
        else:
            # No threshold: start with all indices (may be capped below)
            flat_indices = torch.arange(total, device=device)

        # Step B: cap to effective_cap if defined (top-k selection)
        if effective_cap is not None and flat_indices.numel() > effective_cap:
            selected_vals = log_joint_flat[flat_indices]
            top_vals, rel_idx = torch.topk(selected_vals, k=effective_cap, largest=True)
            flat_indices = flat_indices[rel_idx]
            scores = torch.exp(top_vals)
        else:
            scores = torch.exp(log_joint_flat[flat_indices])

        # Map flat indices to local coordinates
        ER = R * E
        h_local = flat_indices // ER
        rem = flat_indices % ER
        r_local = rem // E
        t_local = rem % E

        # Map local back to original IDs
        heads_tensor = entities[h_local]
        rels_tensor = original_relations[r_local]
        tails_tensor = entities[t_local]
        candidates = torch.stack([heads_tensor, rels_tensor, tails_tensor], dim=1)

        return candidates, scores