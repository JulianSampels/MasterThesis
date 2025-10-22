"""
Scalable routines for training and finetuning a PathER model.

"""

import logging
from functools import partial

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_scatter


from .pathe_ranking_metrics import (RelationHitsAtKUniqueHeads, RelationMRRTriples, RelationMRRTuples, RelationHitsAtKTriples, RelationHitsAtKTuples,
                                   EntityMRRTriples, EntityHitsAtKTriples, CandidateMRRPerSampleFiltered, 
                                   CandidateHitsAtKPerSampleFiltered, CandidateRecallAtKPerGroup, CandidateRecallAtKTotal, RelationMRRUniqueHeads, TailHitsAtKTuples, TailMRRTuples)

from .pather_models import PathEModelTuples


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
                 optimiser="adam", scheduler="none", lrate=1e-3, momentum=0,
                 weight_decay=0, class_weights=None, label_smoothing=0.0,
                 train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                 val_num_negatives=0, full_test=False, max_ppt=None,
                 margin=10, nssa_alpha=1, lp_loss_fn="nssa",
                 loss_weight: float = 0.5, scheduler_monitor="valid_total_loss", scheduler_patience=5, **hparams):
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
        self.lr = lrate  # initial LR
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_patience = scheduler_patience
        self.use_manual_optimization = False
        self.link_head_detached = False
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
            **{f"recall@{k}_total": CandidateRecallAtKTotal(k) for k in self.cand_topk},
        })
        self.cand_metrics_test = nn.ModuleDict({
            "mrr": CandidateMRRPerSampleFiltered(),
            **{f"hits@{k}": CandidateHitsAtKPerSampleFiltered(k) for k in self.cand_topk},
            **{f"recall@{k}_perGroup": CandidateRecallAtKPerGroup(k) for k in self.cand_topk},
            **{f"recall@{k}_total": CandidateRecallAtKTotal(k) for k in self.cand_topk},
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

        # self.save_hyperparameters(ignore=['pathe_model', 'global_head_tail_adjacency', 'train_head_tail_adjacency', 'val_head_tail_adjacency', 'test_head_tail_adjacency', 'filtration_dict'])

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
        self.val_relationMRR(triples=triples, scores=logits)
        self.val_relationHitsAt1(triples=triples, scores=logits)
        self.val_relationHitsAt3(triples=triples, scores=logits)
        self.val_relationHitsAt5(triples=triples, scores=logits)
        self.val_relationHitsAt10(triples=triples, scores=logits)
        self.log("valid_mrr", self.val_relationMRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_hits1", self.val_relationHitsAt1, on_step=False, on_epoch=True)
        self.log("valid_hits3", self.val_relationHitsAt3, on_step=False, on_epoch=True)
        self.log("valid_hits5", self.val_relationHitsAt5, on_step=False, on_epoch=True)
        self.log("valid_hits10", self.val_relationHitsAt10, on_step=False, on_epoch=True)

    def calculate_and_log_test_relation_metrics(self, triples, logits):
        self.test_relationMRR(triples=triples, scores=logits)
        self.test_relationHitsAt1(triples=triples, scores=logits)
        self.test_relationHitsAt3(triples=triples, scores=logits)
        self.test_relationHitsAt5(triples=triples, scores=logits)
        self.test_relationHitsAt10(triples=triples, scores=logits)
        self.log("test_mrr", self.test_relationMRR, on_step=False, on_epoch=True)
        self.log("test_hits1", self.test_relationHitsAt1, on_step=False, on_epoch=True)
        self.log("test_hits3", self.test_relationHitsAt3, on_step=False, on_epoch=True)
        self.log("test_hits5", self.test_relationHitsAt5, on_step=False, on_epoch=True)
        self.log("test_hits10", self.test_relationHitsAt10, on_step=False, on_epoch=True)

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
          - {split}_link_recall@{k}_total
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

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.val_sub_batch_size is None:
            logits_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, logits_lp = self.sub_batch(batch=batch, sub_batch_size=self.val_sub_batch_size)

        targets: torch.Tensor = batch["target"] - 2
        triples: torch.Tensor = batch['ori_triple']


        # Candidate mode if labels present
        if "lp_labels" in batch:
            lp_labels = batch["lp_labels"]
            lp_weights = batch.get("lp_weights")
            lp_groups = batch.get("lp_group_ids")

            lp_loss = self.calculate_lp_bce(logits_lp, labels=lp_labels, sample_weights=lp_weights, num_negatives=None)
            self.log("valid_lp_loss", lp_loss, prog_bar=True)
            self.log("valid_total_loss", lp_loss)

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
        else:
            # RP loss (only make sense if not labels mode as this works without labels and instead with num_negatives)
            rp_loss = self.compute_rp_loss(logits_rp, targets, self.val_num_negatives)
            # self.log("valid_rp_loss", rp_loss, prog_bar=True)

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

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        print()
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

                # Relation metrics (don't work with labels mode)
                # self.calculate_and_log_val_relation_metrics(triples_rp, logits_rp)

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

        # Step schedulers if in manual optimization mode
        if self.use_manual_optimization and self.scheduler == "reduce_on_plateau":
            for scheduler in self.lr_schedulers():
                if self.scheduler_monitor in self.trainer.callback_metrics:
                    scheduler.step(self.trainer.callback_metrics[self.scheduler_monitor])
                    new_lr = scheduler.optimizer.param_groups[0]['lr']
                    self.log("lr", new_lr, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        if self.test_sub_batch_size is None:
            logits_rp, logits_lp = self.model_forward(batch=batch)
        else:
            logits_rp, logits_lp = self.sub_batch(batch=batch, sub_batch_size=self.test_sub_batch_size)

        targets = batch["target"] - 2
        triples = batch['ori_triple']

        # Candidate mode per-batch metrics
        if "lp_labels" in batch:
            lp_labels = batch["lp_labels"]
            lp_weights = batch.get("lp_weights")
            lp_groups = batch.get("lp_group_ids")

            lp_loss = self.calculate_lp_bce(logits_lp, labels=lp_labels, sample_weights=lp_weights, num_negatives=None)
            self.log("test_lp_loss", lp_loss, prog_bar=True)
            # self.log("test_total_loss", (self.loss_weight * lp_loss) + ((1.0 - self.loss_weight) * rp_loss),
            #          prog_bar=self.loss_weight not in [0.0, 1.0])

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
        else:
            # RP loss (only make sense if not labels mode as this works without labels and instead with num_negatives)
            rp_loss = self.compute_rp_loss(logits_rp, targets, self.test_num_negatives)
            self.log("test_rp_loss", rp_loss, prog_bar=True)
        
        # Legacy test path
        if self.test_num_negatives > 0:
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

    @torch.no_grad()
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
        if self.use_manual_optimization:
            if self.optimiser == 'adam':
                # Return separate optimizers for relation and link prediction
                relation_params = []
                tail_params = []
                shared_params = []
                
                # Separate parameters based on their role
                for name, param in self.model.named_parameters():
                    if 'relpredict_head_avg' in name:
                        relation_params.append(param)
                        print(f"Relation param: {name}")
                    elif 'tail_predict_head' in name:
                        tail_params.append(param)
                        print(f"Tail param: {name}")
                    else:
                        shared_params.append(param)
                
                relation_optimizer = torch.optim.Adam(
                    shared_params + relation_params, 
                    lr=self.lr, 
                    weight_decay=self.weight_decay
                )
                
                link_optimizer = torch.optim.Adam(
                    (shared_params + tail_params) if not self.link_head_detached else tail_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )
            else:
                raise ValueError(f"Optimizer `{self.optimiser}` not supported for manual optimization.")
            
            # Create schedulers if needed
            optimizers_and_schedulers = [
                {"optimizer": relation_optimizer},
                {"optimizer": link_optimizer}
            ]
            if self.scheduler == "reduce_on_plateau":
                relation_scheduler = ReduceLROnPlateau(
                    optimizer=relation_optimizer,
                    mode='min' if self.scheduler_monitor.endswith("loss") else "max",
                    factor=0.1,
                    patience=self.scheduler_patience,
                )
                link_scheduler = ReduceLROnPlateau(
                    optimizer=link_optimizer,
                    mode='min' if self.scheduler_monitor.endswith("loss") else "max",
                    factor=0.1,
                    patience=self.scheduler_patience,
                )
                optimizers_and_schedulers[0]["lr_scheduler"] = {
                    "scheduler": relation_scheduler,
                    "monitor": self.scheduler_monitor,
                }
                optimizers_and_schedulers[1]["lr_scheduler"] = {
                    "scheduler": link_scheduler,
                    "monitor": self.scheduler_monitor,
                }
            elif self.scheduler != "none":
                raise ValueError(f"Scheduler `{self.scheduler}` not supported.")
            
            return optimizers_and_schedulers
        else:
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
            if self.scheduler == "reduce_on_plateau":
                scheduler = ReduceLROnPlateau(
                    optimizer=optimiser,
                    mode='min' if self.scheduler_monitor.endswith("loss") else "max",
                    factor=0.1,
                    patience=self.scheduler_patience,
                )
                opt_dict["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_monitor,
                }
            elif self.scheduler != "none":
                raise ValueError(f"Scheduler `{self.scheduler}` not supported.")
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
                pathe_model: PathEModelTuples, filtration_dict, 
                train_head_tail_adjacency: torch.Tensor = None,
                val_head_tail_adjacency: torch.Tensor = None,
                test_head_tail_adjacency: torch.Tensor = None,
                num_negatives=0,
                optimiser="adam", scheduler="none", lrate=1e-3, momentum=0,
                weight_decay=0, class_weights=None, label_smoothing=0.0,
                train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                val_num_negatives=0, full_test=False, max_ppt=None, accumulate_gradient=1.0, 
                margin=10, nssa_alpha=1, lp_loss_fn="nssa", link_head_detached: bool = True, use_manual_optimization: bool = False,
                loss_weight: float = 0.5,
                **hparams):
        super().__init__(pathe_model, filtration_dict, num_negatives,
                         optimiser, scheduler, lrate, momentum, weight_decay,
                         class_weights, label_smoothing,
                         train_sub_batch=train_sub_batch,
                         val_sub_batch=val_sub_batch,
                         test_sub_batch=test_sub_batch,
                         val_num_negatives=val_num_negatives,
                         full_test=full_test, max_ppt=max_ppt,
                         margin=margin, nssa_alpha=nssa_alpha,
                         lp_loss_fn=lp_loss_fn, loss_weight=loss_weight, **hparams)
        
        # Ensure sane integer gradient accumulation
        self.accumulate_gradient = max(1, int(round(accumulate_gradient)))
        self.use_manual_optimization = use_manual_optimization
        self.link_head_detached = link_head_detached

        # Set automatic optimization based on parameter
        self.automatic_optimization = not self.use_manual_optimization

        # Store head-tail adjacency matrices as buffers
        if train_head_tail_adjacency is not None:
            self.register_buffer('train_head_tail_adjacency', train_head_tail_adjacency)
        else:
            self.train_head_tail_adjacency = None

        if val_head_tail_adjacency is not None:
            self.register_buffer('val_head_tail_adjacency', val_head_tail_adjacency)
        else:
            self.val_head_tail_adjacency = None

        if test_head_tail_adjacency is not None:
            self.register_buffer('test_head_tail_adjacency', test_head_tail_adjacency)
        else:
            self.test_head_tail_adjacency = None

        # Compute global after registration (buffers are on device now)
        if self.train_head_tail_adjacency is not None and self.val_head_tail_adjacency is not None and self.test_head_tail_adjacency is not None:
            self.global_head_tail_adjacency = self.train_head_tail_adjacency | self.val_head_tail_adjacency | self.test_head_tail_adjacency
        else:
            self.global_head_tail_adjacency = None

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

        # Tail metrics with global adjacency
        self.val_tailMRR = TailMRRTuples(filter_global_adjacency=self.global_head_tail_adjacency)
        self.val_tailHitsAt1 = TailHitsAtKTuples(k=1, filter_global_adjacency=self.global_head_tail_adjacency)
        self.val_tailHitsAt3 = TailHitsAtKTuples(k=3, filter_global_adjacency=self.global_head_tail_adjacency)
        self.val_tailHitsAt5 = TailHitsAtKTuples(k=5, filter_global_adjacency=self.global_head_tail_adjacency)
        self.val_tailHitsAt10 = TailHitsAtKTuples(k=10, filter_global_adjacency=self.global_head_tail_adjacency)

        self.test_tailMRR = TailMRRTuples(filter_global_adjacency=self.global_head_tail_adjacency)
        self.test_tailHitsAt1 = TailHitsAtKTuples(k=1, filter_global_adjacency=self.global_head_tail_adjacency)
        self.test_tailHitsAt3 = TailHitsAtKTuples(k=3, filter_global_adjacency=self.global_head_tail_adjacency)
        self.test_tailHitsAt5 = TailHitsAtKTuples(k=5, filter_global_adjacency=self.global_head_tail_adjacency)
        self.test_tailHitsAt10 = TailHitsAtKTuples(k=10, filter_global_adjacency=self.global_head_tail_adjacency)

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

        # self.save_hyperparameters(ignore=['pathe_model', 'global_head_tail_adjacency', 'train_head_tail_adjacency', 'val_head_tail_adjacency', 'test_head_tail_adjacency', 'filtration_dict'])

    def compute_rp_loss(self, logits, targets, num_negatives):
        loss_rp = self.rp_loss_fn(logits, targets)
        assert(loss_rp.size(0) % (num_negatives + 1) == 0), "Incompatible loss size and negative sample sizes probably something when wrong with the batch size when generating negatives!"
        loss_rp = torch.mean(loss_rp[torch.arange(0, loss_rp.size()[0], num_negatives + 1)])
        return loss_rp

    def compute_lp_loss(self, logits, num_negatives):
        # legacy path not used in tuples tail training
        loss_lp = self.lp_loss_fn(logits, num_negatives)
        return loss_lp
    
    def compute_tail_bce_loss(self, logits_tail: torch.Tensor, heads: torch.Tensor, adjacency_matrix: torch.Tensor):
        if adjacency_matrix is None:
            raise ValueError("An adjacency matrix must be provided for on-the-fly loss calculation.")
        
        # Generate labels and weights on the fly using the provided adjacency matrix
        tail_labels = adjacency_matrix[heads.to(adjacency_matrix.device)].to(device=self.device, dtype=torch.float32, non_blocking=True)
        
        pos_counts = tail_labels.sum(dim=1)
        neg_counts = tail_labels.shape[1] - pos_counts
        
        w_pos = 0.5 / pos_counts.clamp_min(1.0)
        w_neg = 0.5 / neg_counts.clamp_min(1.0)
        
        tail_weights = torch.where(tail_labels > 0.5, w_pos.unsqueeze(1), w_neg.unsqueeze(1))
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits_tail, tail_labels, weight=tail_weights, reduction='sum')
        return loss / tail_weights.sum().clamp_min(1.0)

    def training_step(self, batch, batch_idx):
        # Forward pass
        logits_rp1, logits_rp2, logits_tail1, logits_tail2 = self.model_forward(batch)
        targets = batch["target"] - 2
        tuples = batch["ori_triple"]                    # (N, 2) => (h, r)
        heads = tuples[:, 0].to(self.device)

        # Losses (unscaled)
        rp_loss_unscaled = self.compute_rp_loss(logits_rp1, targets, self.train_num_negatives)
        tail_loss_unscaled = self.compute_tail_bce_loss(logits_tail1, heads, self.train_head_tail_adjacency)

        if not self.use_manual_optimization:
            total = (1.0 - self.loss_weight) * rp_loss_unscaled + self.loss_weight * tail_loss_unscaled
            self.log("train_rp_loss", rp_loss_unscaled, prog_bar=True)
            self.log("train_tp_loss", tail_loss_unscaled, prog_bar=True)
            self.log("train_total_loss", total, prog_bar=True)
            return total

        # Manual optimization mode
        relation_opt, tail_opt = self.optimizers()
        scale = 1.0 / self.accumulate_gradient
        rp_loss = rp_loss_unscaled * scale
        tail_loss = tail_loss_unscaled * scale

        # Backward passes
        self.toggle_optimizer(optimizer=relation_opt, optimizer_idx=0)
        self.manual_backward(rp_loss, retain_graph=(not self.link_head_detached))
        self.untoggle_optimizer(optimizer_idx=0)

        if tail_loss.requires_grad:
            self.toggle_optimizer(optimizer=tail_opt, optimizer_idx=1)
            self.manual_backward(tail_loss, retain_graph=False)
            self.untoggle_optimizer(optimizer_idx=1)

        # Step and zero grads
        is_boundary = ((batch_idx + 1) % self.accumulate_gradient) == 0
        total_batches = getattr(self.trainer, "num_training_batches", None)
        is_last = (total_batches is not None) and ((batch_idx + 1) == int(total_batches))
        if is_boundary or is_last:
            self.clip_gradients(relation_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            relation_opt.step(); relation_opt.zero_grad()
            self.clip_gradients(tail_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            tail_opt.step(); tail_opt.zero_grad()

        # Logging unscaled losses
        self.log("train_rp_loss", rp_loss_unscaled, prog_bar=True)
        self.log("train_tp_loss", tail_loss_unscaled, prog_bar=True)
        self.log("train_total_loss", rp_loss_unscaled + tail_loss_unscaled)
        return {"rp_loss": rp_loss_unscaled, "tp_loss": tail_loss_unscaled}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Forward pass
        logits_rp1, logits_rp2, logits_tail1, logits_tail2 = self.model_forward(batch)
        targets = batch["target"] - 2
        tuples = batch['ori_triple']
        heads = tuples[:, 0].to(self.device)

        # Losses
        rp_loss = self.compute_rp_loss(logits_rp2, targets, self.val_num_negatives)
        tp_loss = self.compute_tail_bce_loss(logits_tail2, heads, self.val_head_tail_adjacency)

        # Logging losses
        self.log("valid_rp_loss", rp_loss, prog_bar=True)
        self.log("valid_tp_loss", tp_loss, prog_bar=True)
        self.log("valid_total_loss", (1.0 - self.loss_weight) * rp_loss + self.loss_weight * tp_loss)

        # --- Relation prediction metrics (only positives) --------------------
        pos_indices = torch.arange(0, tuples.size(0), self.val_num_negatives + 1)
        tuples_only_positives = tuples[pos_indices]
        logits_rp_only_positives = logits_rp2[pos_indices]

        # Accumulate validation data for epoch-end processing
        if not hasattr(self, "_val_acc"):
            assert(batch_idx == 0), "Somehow accumulating validation data was not deleted in validation_epoch_end() after last epoch. Risking incorrect metrics!"
            self._val_acc = {"tuples_rp": [], "tuples_tp": [], "logits_rp": [], "logits_tp": []}
        self._val_acc["tuples_rp"].append(tuples_only_positives.detach().cpu())
        self._val_acc["tuples_tp"].append(tuples.detach().cpu())
        self._val_acc["logits_rp"].append(logits_rp_only_positives.detach().cpu())
        self._val_acc["logits_tp"].append(logits_tail2.detach().cpu())

        return {"rp_loss": rp_loss, "tp_loss": tp_loss}

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        if not hasattr(self, "_val_acc"):
            logger.warning("validation_epoch_end() called without any accumulated validation data!")
            return
        # Only relation metrics for tuples here
        triples_rp = torch.cat(self._val_acc["tuples_rp"], dim=0)
        logits_rp = torch.cat(self._val_acc["logits_rp"], dim=0)
        assert(logits_rp.size(0) == triples_rp.size(0))
        self.calculate_and_log_val_relation_metrics(triples_rp, logits_rp)


        # --- NEW: Tail metrics -------------------------------------------
        # All (possibly duplicated) heads with tail logits
        tuples_all = torch.cat(self._val_acc["tuples_tp"], dim=0)          # (N, 2) (h, r)
        logits_tail_all = torch.cat(self._val_acc["logits_tp"], dim=0)     # (N, E)
        heads_all = tuples_all[:, 0]                                       # (N,)

        # Group by head: aggregate (mean) scores across multiple (h,r) rows
        unique_heads, inverse = heads_all.unique(return_inverse=True, sorted=False)
        # If torch_scatter available (already imported), use it:
        scores_agg = torch_scatter.scatter_mean(logits_tail_all, inverse, dim=0)
        # Aggregate eval labels per unique head (take first since identical)
        eval_labels_agg = self.val_head_tail_adjacency[unique_heads].to(torch.float32)

        # Update tail metrics with eval labels
        self.val_tailMRR.update(unique_heads, scores_agg, eval_labels_agg)
        self.val_tailHitsAt1.update(unique_heads, scores_agg, eval_labels_agg)
        self.val_tailHitsAt3.update(unique_heads, scores_agg, eval_labels_agg)
        self.val_tailHitsAt5.update(unique_heads, scores_agg, eval_labels_agg)
        self.val_tailHitsAt10.update(unique_heads, scores_agg, eval_labels_agg)

        # Log tail metrics
        self.log("valid_tail_mrr", self.val_tailMRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_tail_hits1", self.val_tailHitsAt1, on_step=False, on_epoch=True)
        self.log("valid_tail_hits3", self.val_tailHitsAt3, on_step=False, on_epoch=True)
        self.log("valid_tail_hits5", self.val_tailHitsAt5, on_step=False, on_epoch=True)
        self.log("valid_tail_hits10", self.val_tailHitsAt10, on_step=False, on_epoch=True)

        # Step schedulers if in manual optimization mode
        if self.use_manual_optimization and self.scheduler == "reduce_on_plateau":
            for scheduler in self.lr_schedulers():
                if self.scheduler_monitor in self.trainer.callback_metrics:
                    scheduler.step(self.trainer.callback_metrics[self.scheduler_monitor])
                    new_lr = scheduler.optimizer.param_groups[0]['lr']
                    self.log("lr", new_lr, on_epoch=True, prog_bar=True)

        # Cleanup
        del self._val_acc

    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        logits_rp, logits_tp = self.pathe_forward_step_tuples(batch)
        tuples = batch['ori_triple']  # (num_samples, 2) -> (h, r)
        return {
            "tuples": tuples.detach().cpu(),
            "scores_rp": logits_rp.detach().cpu(),
            "scores_tp": logits_tp.detach().cpu()
        }
    
    
    # quite similar to validation_step but with different logging names and metric calculation functions
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # Forward pass
        logits_rp1, logits_rp2, logits_tail1, logits_tail2 = self.model_forward(batch)
        targets = batch["target"] - 2
        tuples = batch['ori_triple']
        heads = tuples[:, 0].to(self.device)

        # Compute losses
        rp_loss = self.compute_rp_loss(logits_rp2, targets, self.test_num_negatives)
        tp_loss = self.compute_tail_bce_loss(logits_tail2, heads, self.test_head_tail_adjacency)

        # Logging losses
        self.log("test_rp_loss", rp_loss, prog_bar=True)
        self.log("test_tp_loss", tp_loss, prog_bar=True)
        self.log("test_total_loss", rp_loss + tp_loss)

        # Relation metrics (tuples)
        assert(tuples.size()[0] == logits_rp2.size()[0])
        logits_rp_only_positives = logits_rp2[torch.arange(0, logits_rp2.size()[0], self.test_num_negatives + 1)]
        tuples_only_positives = tuples[torch.arange(0, tuples.size()[0], self.test_num_negatives + 1)]

        if not hasattr(self, "_test_acc"):
            assert batch_idx == 0, "Accumulation not cleared from previous test epoch."
            self._test_acc = {"tuples_rp": [], "tuples_tp": [], "logits_rp": [], "logits_tp": []}

        self._test_acc["tuples_rp"].append(tuples_only_positives.detach().cpu())
        self._test_acc["logits_rp"].append(logits_rp_only_positives.detach().cpu())
        self._test_acc["tuples_tp"].append(tuples.detach().cpu())
        self._test_acc["logits_tp"].append(logits_tail2.detach().cpu())

        return {"rp_loss": rp_loss, "tp_loss": tp_loss}

        # self.calculate_and_log_test_relation_metrics(tuples_only_positives, logits_rp_only_positives)

    @torch.no_grad()
    def on_test_epoch_end(self):
        if not hasattr(self, "_test_acc"):
            return
        # Relation metrics
        tuples_rp = torch.cat(self._test_acc["tuples_rp"], dim=0)
        logits_rp = torch.cat(self._test_acc["logits_rp"], dim=0)
        assert logits_rp.size(0) == tuples_rp.size(0)
        self.calculate_and_log_test_relation_metrics(tuples_rp, logits_rp)

        # Tail metrics
        tuples_all = torch.cat(self._test_acc["tuples_tp"], dim=0)
        logits_tail_all = torch.cat(self._test_acc["logits_tp"], dim=0)
        heads_all = tuples_all[:, 0]
        unique_heads, inverse = heads_all.unique(return_inverse=True, sorted=False)
        scores_agg = torch_scatter.scatter_mean(logits_tail_all, inverse, dim=0)
        # Aggregate eval labels per unique head
        eval_labels_agg = self.test_head_tail_adjacency[unique_heads].to(torch.float32)

        self.test_tailMRR.update(unique_heads, scores_agg, eval_labels_agg)
        self.test_tailHitsAt1.update(unique_heads, scores_agg, eval_labels_agg)
        self.test_tailHitsAt3.update(unique_heads, scores_agg, eval_labels_agg)
        self.test_tailHitsAt5.update(unique_heads, scores_agg, eval_labels_agg)
        self.test_tailHitsAt10.update(unique_heads, scores_agg, eval_labels_agg)

        self.log("test_tail_mrr", self.test_tailMRR, on_step=False, on_epoch=True)
        self.log("test_tail_hits1", self.test_tailHitsAt1, on_step=False, on_epoch=True)
        self.log("test_tail_hits3", self.test_tailHitsAt3, on_step=False, on_epoch=True)
        self.log("test_tail_hits5", self.test_tailHitsAt5, on_step=False, on_epoch=True)
        self.log("test_tail_hits10", self.test_tailHitsAt10, on_step=False, on_epoch=True)

        del self._test_acc

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
        # targets = batch["target"] - 2  # no PAD and MSK in the model output
        # FIXME this is going to change with the actual head-tail idxs from dataset
        head_idxs = head_idxs * 2
        ppt = batch["ppt"]

        logits_rp, logits_tp, logits_rcount, logits_tcount = self.model(
            ent_paths=ent_paths,
            rel_paths=rel_paths,
            head_idxs=head_idxs,
            ppt=ppt,
            pos=pos,
            entity_origin=entity_origin,
            detach_link_head=self.link_head_detached
        )
        # loss_rp = self.criterion(logits_rp, targets)
        return logits_rp, logits_tp, logits_rcount, logits_tcount

class PathEModelWrapperUniqueHeads(PathEModelWrapperTuples):
    def __init__(self, 
                pathe_model, filtration_dict,
                train_head_tail_adjacency=None,
                val_head_tail_adjacency=None,
                test_head_tail_adjacency=None,
                num_negatives=0,
                optimiser="adam", scheduler="none", lrate=1e-3, momentum=0,
                weight_decay=0, class_weights=None, label_smoothing=0.0,
                train_sub_batch=None, val_sub_batch=None, test_sub_batch=None,
                val_num_negatives=0, full_test=False, max_ppt=None, accumulate_gradient=1.0, 
                margin=10, nssa_alpha=1, lp_loss_fn="nssa", link_head_detached=True, use_manual_optimization=False,
                loss_weight=0.5,
                train_relation_count_matrix=None,
                val_relation_count_matrix=None,
                test_relation_count_matrix=None,
                phase1_loss_fn='bce',
                **kwargs):
        super().__init__(pathe_model, filtration_dict, train_head_tail_adjacency, val_head_tail_adjacency, test_head_tail_adjacency, num_negatives,
                         optimiser, scheduler, lrate, momentum, weight_decay,
                         class_weights, label_smoothing,
                         train_sub_batch=train_sub_batch,
                         val_sub_batch=val_sub_batch,
                         test_sub_batch=test_sub_batch,
                         val_num_negatives=val_num_negatives,
                         full_test=full_test, max_ppt=max_ppt,
                         accumulate_gradient=accumulate_gradient, margin=margin, nssa_alpha=nssa_alpha,
                         lp_loss_fn=lp_loss_fn, link_head_detached=link_head_detached, use_manual_optimization=use_manual_optimization,
                         loss_weight=loss_weight, **kwargs)
        self.phase1_loss_fn = phase1_loss_fn
        
        # Store relation count matrices as buffers
        if train_relation_count_matrix is not None:
            self.register_buffer('train_relation_count_matrix', train_relation_count_matrix)
        else:
            self.train_relation_count_matrix = None

        if val_relation_count_matrix is not None:
            self.register_buffer('val_relation_count_matrix', val_relation_count_matrix)
        else:
            self.val_relation_count_matrix = None

        if test_relation_count_matrix is not None:
            self.register_buffer('test_relation_count_matrix', test_relation_count_matrix)
        else:
            self.test_relation_count_matrix = None

        # Compute global after registration
        if self.train_relation_count_matrix is not None and self.val_relation_count_matrix is not None and self.test_relation_count_matrix is not None:
            self.global_relation_count_matrix = self.train_relation_count_matrix | self.val_relation_count_matrix | self.test_relation_count_matrix
        else:
            self.global_relation_count_matrix = None
        
        
        # Override relation metrics for unique heads
        self.val_relationMRR = RelationMRRUniqueHeads(self.global_relation_count_matrix)
        self.val_relationHitsAt1 = RelationHitsAtKUniqueHeads(k=1, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.val_relationHitsAt3 = RelationHitsAtKUniqueHeads(k=3, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.val_relationHitsAt5 = RelationHitsAtKUniqueHeads(k=5, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.val_relationHitsAt10 = RelationHitsAtKUniqueHeads(k=10, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        
        self.test_relationMRR = RelationMRRUniqueHeads(self.global_relation_count_matrix)
        self.test_relationHitsAt1 = RelationHitsAtKUniqueHeads(k=1, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.test_relationHitsAt3 = RelationHitsAtKUniqueHeads(k=3, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.test_relationHitsAt5 = RelationHitsAtKUniqueHeads(k=5, filter_global_relation_count_matrix=self.global_relation_count_matrix)
        self.test_relationHitsAt10 = RelationHitsAtKUniqueHeads(k=10, filter_global_relation_count_matrix=self.global_relation_count_matrix)
    
    def compute_rp_bce_loss(self, logits, relation_count_matrix: torch.Tensor):
        """
        Compute multi-label BCE loss for relation prediction.
        logits: (batch_size, num_relations)
        relation_count_matrix: (batch_size, num_relations) indexed matrix
        """
        batch_size, num_relations = logits.shape
        
        # Create binary labels: 1 if used (count > 0), 0 otherwise
        targets = (relation_count_matrix > 0).float()

        # weights = torch.ones_like(relation_count_matrix)
        # pos_counts = relation_count_matrix.sum(dim=1)
        # neg_counts = relation_count_matrix.shape[1] - pos_counts
        # w_pos = 0.5 / pos_counts.clamp_min(1.0)
        # w_neg = 0.5 / neg_counts.clamp_min(1.0)
        # weights = torch.where(relation_count_matrix > 0.5, w_pos.unsqueeze(1), w_neg.unsqueeze(1))
        # weights = torch.ones_like(targets)  # uncomment to disable weighting

        # Weights: use the count for positives, 1.0 for negatives
        # weights = torch.where(targets > 0.5, relation_count_matrix, 1.0)

        # Weights: use the count for positives, and for negatives, distribute the total positive weight equally
        num_pos = targets.sum(dim=1)  # (batch_size,)
        num_neg = (targets <= 0.5).sum(dim=1)  # (batch_size,)
        weight_neg = 1 / num_neg.clamp_min(1.0)  # (batch_size,)
        weight_pos = 1 / num_pos.clamp_min(1.0)  # (batch_size,)
        weights = torch.where(targets > 0.5, weight_pos.unsqueeze(1).expand_as(targets), weight_neg.unsqueeze(1).expand_as(targets))

        # Apply label smoothing
        smoothed_targets = targets * (1 - self.label_smoothing) + (1 - targets) * self.label_smoothing

        # Use BCE with logits, averaged over all elements (batch * relations)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, smoothed_targets, weight=weights, reduction='sum')
        loss /= weights.sum().clamp_min(1.0)
        return loss

    def compute_tail_bce_loss(self, logits_tail: torch.Tensor, entity_count_matrix: torch.Tensor):
        if entity_count_matrix is None:
            raise ValueError("An adjacency matrix must be provided for on-the-fly loss calculation.")
        
        # Generate labels and weights on the fly using the provided adjacency matrix
        # Create binary labels: 1 if used (count > 0), 0 otherwise
        targets = (entity_count_matrix > 0).float()

        # weights = torch.ones_like(targets)
        # pos_counts = targets.sum(dim=1)
        # neg_counts = targets.shape[1] - pos_counts
        # w_pos = 0.5 / pos_counts.clamp_min(1.0)
        # w_neg = 0.5 / neg_counts.clamp_min(1.0)
        # tail_weights = torch.where(targets > 0.5, w_pos.unsqueeze(1), w_neg.unsqueeze(1))
        # tail_weights = torch.ones_like(targets)  # uncomment to disable weighting

        # Weights: use the count for positives, 1.0 for negatives
        # tail_weights = torch.where(targets > 0.5, entity_count_matrix, 1.0)

        # Weights: use the count for positives, and for negatives, distribute the total positive weight equally
        sum_pos = entity_count_matrix.sum(dim=1)  # (batch_size,)
        num_neg = (targets <= 0.5).sum(dim=1)  # (batch_size,)
        weight_neg = sum_pos / num_neg.clamp_min(1.0)  # (batch_size,)
        tail_weights = torch.where(targets > 0.5, entity_count_matrix, weight_neg.unsqueeze(1).expand_as(entity_count_matrix))

        # Apply label smoothing
        smoothed_targets = targets * (1 - self.label_smoothing) + (1 - targets) * self.label_smoothing
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits_tail, smoothed_targets, weight=tail_weights, reduction='sum')
        loss /= tail_weights.sum().clamp_min(1.0)
        return loss

    def compute_rp_poisson_loss(self, logits: torch.Tensor, relation_count_matrix: torch.Tensor):
        """
        Compute Poisson NLL loss for relation prediction.
        """
        loss = torch.nn.functional.poisson_nll_loss(logits, relation_count_matrix, log_input=True, full=True, reduction='sum')
        loss /= relation_count_matrix.numel()
        return loss

    def compute_tail_poisson_loss(self, logits_tail: torch.Tensor, entity_count_matrix: torch.Tensor):
        """
        Compute weighted Poisson NLL loss for tail prediction.
        Weights are automatically set based on zero vs. non-zero counts for balance.
        """
        if entity_count_matrix is None:
            raise ValueError("An entity count matrix must be provided for Poisson loss calculation.")
        
        # Compute per-element Poisson NLL (reduction='none')
        # loss_per_element = torch.nn.functional.poisson_nll_loss(predictions, entity_count_matrix, log_input=True, reduction='none')
        
        # Automatic weighting: balance zeros vs. non-zeros
        # num_entities = entity_count_matrix.shape[1]
        # num_zeros = (entity_count_matrix == 0).sum(dim=1)
        # num_non_zeros = num_entities - num_zeros
        
        # Inverse frequency weights (avoid division by zero)
        # zero_weights = num_entities / num_zeros.clamp_min(1.0)
        # non_zero_weights = num_entities / num_non_zeros.clamp_min(1.0)

        # weights = torch.where(entity_count_matrix == 0, zero_weights.unsqueeze(1), non_zero_weights.unsqueeze(1))
        # weights = torch.where(entity_count_matrix == 0, zero_weights, non_zero_weights)

        # Weighted sum, normalized by total weight (ensures scale stability)
        # weighted_loss = (loss_per_element * weights).sum()
        # total_weight = weights.sum().clamp_min(1.0)
        # loss = weighted_loss / total_weight

        loss = torch.nn.functional.poisson_nll_loss(logits_tail, entity_count_matrix, log_input=True, full=True, reduction='sum')
        loss /= entity_count_matrix.numel()
        return loss

    def compute_tail_hurdle_loss(self, logits1, logits2, entity_count_matrix):
        hurdle_loss = self.compute_tail_bce_loss(logits1, entity_count_matrix)
        poisson_mask = (entity_count_matrix > 0).float()
        if poisson_mask.sum() > 0:
            # Compute per-element Poisson NLL loss on the full tensors
            loss_per_element = torch.nn.functional.poisson_nll_loss(
                logits2, entity_count_matrix, log_input=True, full=True, reduction='none'
            )
            # Mask out losses for non-positive tails (where mask == 0)
            masked_loss = loss_per_element * poisson_mask
            # Sum only the masked losses and normalize by the number of positive elements
            count_loss = masked_loss.sum() / poisson_mask.sum().clamp_min(1.0)
        else:
            count_loss = torch.tensor(0.0, device=self.device)
        # print(f"hurdle_loss: {hurdle_loss}, count_loss: {count_loss}")
        hurdle_loss /= hurdle_loss.detach().clamp_min(1.0)
        count_loss /= count_loss.detach().clamp_min(1.0)
        return hurdle_loss + count_loss

    def compute_rp_hurdle_loss(self, logits1, logits2, relation_count_matrix):
        hurdle_loss = self.compute_rp_bce_loss(logits1, relation_count_matrix)
        poisson_mask = (relation_count_matrix > 0).float()
        if poisson_mask.sum() > 0:
            # Compute per-element Poisson NLL loss on the full tensors
            loss_per_element = torch.nn.functional.poisson_nll_loss(
                logits2, relation_count_matrix, log_input=True, full=True, reduction='none'
            )
            # Mask out losses for non-positive relations (where mask == 0)
            masked_loss = loss_per_element * poisson_mask
            # Sum only the masked losses and normalize by the number of positive elements
            count_loss =  masked_loss.sum() / poisson_mask.sum().clamp_min(1.0)
        else:
            count_loss = torch.tensor(0.0, device=self.device)
        # print(f"hurdle_loss: {hurdle_loss}, count_loss: {count_loss}")
        hurdle_loss /= hurdle_loss.detach().clamp_min(1.0)
        count_loss /= count_loss.detach().clamp_min(1.0)
        return hurdle_loss + count_loss

    def compute_rp_negative_binomial_loss(self, log_mu, log_alpha, relation_count_matrix):
        """
        Compute Negative Binomial NLL loss for relation prediction.
        log_mu: log of the predicted mean (μ)
        log_alpha: log of the predicted dispersion (α)
        relation_count_matrix: target counts (k)
        """
        mu = torch.exp(log_mu)
        alpha = torch.exp(log_alpha)
        k = relation_count_matrix.to(device=self.device, dtype=torch.float32, non_blocking=True)

        # Negative Binomial log-likelihood (numerical stability with lgamma)
        log_likelihood = (
            torch.lgamma(k + alpha)
            - torch.lgamma(k + 1)
            - torch.lgamma(alpha)
            + alpha * torch.log(alpha / (alpha + mu))
            + k * torch.log(mu / (alpha + mu))
        )
        
        # Return negative log-likelihood, averaged over batch
        return -log_likelihood.mean()

    def compute_tail_negative_binomial_loss(self, log_mu_tail, log_alpha_tail, entity_count_matrix):
        """
        Compute Negative Binomial NLL loss for tail prediction.
        log_mu_tail: log of the predicted mean (μ) for tails
        log_alpha_tail: log of the predicted dispersion (α) for tails
        entity_count_matrix: target counts (k) for tails
        """
        if entity_count_matrix is None:
            raise ValueError("An entity count matrix must be provided for Negative Binomial loss calculation.")
        
        mu = torch.exp(log_mu_tail)
        alpha = torch.exp(log_alpha_tail)
        k = entity_count_matrix

        # Negative Binomial log-likelihood
        log_likelihood = (
            torch.lgamma(k + alpha)
            - torch.lgamma(k + 1)
            - torch.lgamma(alpha)
            + alpha * torch.log(alpha / (alpha + mu))
            + k * torch.log(mu / (alpha + mu))
        )
        
        # Return negative log-likelihood, averaged over batch
        return -log_likelihood.mean()

    def compute_phase1_losses(self, logits_rp_bce, logits_rp_poisson, logits_tail_bce, logits_tail_poisson, relation_count_matrix, entity_count_matrix, phase1_loss_fn):
        """
        Centralized function to compute phase 1 losses (relation and tail prediction) based on the loss function type.
        For negative_binomial, reuse logits_rp_bce as log_mu_rp, logits_rp_poisson as log_alpha_rp,
        logits_tail_bce as log_mu_tp, logits_tail_poisson as log_alpha_tp.
        """
        if phase1_loss_fn == 'bce':
            rp_loss = self.compute_rp_bce_loss(logits_rp_bce, relation_count_matrix)
            tp_loss = self.compute_tail_bce_loss(logits_tail_bce, entity_count_matrix)
        elif phase1_loss_fn == 'poisson':
            rp_loss = self.compute_rp_poisson_loss(logits_rp_poisson, relation_count_matrix)
            tp_loss = self.compute_tail_poisson_loss(logits_tail_poisson, entity_count_matrix)
        elif phase1_loss_fn == 'negative_binomial':
            rp_loss = self.compute_rp_negative_binomial_loss(logits_rp_bce, logits_rp_poisson, relation_count_matrix)
            tp_loss = self.compute_tail_negative_binomial_loss(logits_tail_bce, logits_tail_poisson, entity_count_matrix)
        elif phase1_loss_fn == 'hurdletail':
            rp_loss = self.compute_rp_poisson_loss(logits_rp_poisson, relation_count_matrix)
            tp_loss = self.compute_tail_hurdle_loss(logits_tail_bce, logits_tail_poisson, entity_count_matrix)
        elif phase1_loss_fn == 'hurdlerelation':
            rp_loss = self.compute_rp_hurdle_loss(logits_rp_bce, logits_rp_poisson, relation_count_matrix)
            tp_loss = self.compute_tail_poisson_loss(logits_tail_poisson, entity_count_matrix)
        elif phase1_loss_fn == 'hurdleboth':
            rp_loss = self.compute_rp_hurdle_loss(logits_rp_bce, logits_rp_poisson, relation_count_matrix)
            tp_loss = self.compute_tail_hurdle_loss(logits_tail_bce, logits_tail_poisson, entity_count_matrix)
        else:
            raise ValueError(f"Invalid phase1_loss_fn: {phase1_loss_fn}. Must be 'bce', 'poisson', 'hurdle', 'hurdlerelation', or 'hurdleboth'.")
        return rp_loss, tp_loss
    
    def training_step(self, batch, batch_idx):
        logits_rp1, logits_tail1, logits_rp2, logits_tail2 = self.model_forward(batch)
        heads = batch["heads"]
        relation_count_matrix = self.train_relation_count_matrix[heads.to(self.train_relation_count_matrix.device)]
        entity_count_matrix = self.train_head_tail_adjacency[heads.to(self.train_head_tail_adjacency.device)]

        # Losses
        rp_loss_unscaled, tp_loss_unscaled = self.compute_phase1_losses(logits_rp1, logits_rp2, logits_tail1, logits_tail2, relation_count_matrix, entity_count_matrix, self.phase1_loss_fn)

        if not self.use_manual_optimization:
            total_loss = (1.0 - self.loss_weight) * rp_loss_unscaled + self.loss_weight * tp_loss_unscaled
            self.log("train_rp_loss", rp_loss_unscaled, prog_bar=True)
            self.log("train_tp_loss", tp_loss_unscaled, prog_bar=True)
            self.log("train_total_loss", total_loss)
            return total_loss

        # Manual optimization mode
        # Determine if we are at gradient accumulation boundary
        total_batches = int(getattr(self.trainer, "num_training_batches", None))
        if total_batches is not None:
            remaining_including_this = total_batches - batch_idx
            if remaining_including_this <= self.accumulate_gradient:
                remainder = total_batches % self.accumulate_gradient
                num_in_accum = remainder if remainder != 0 else self.accumulate_gradient
            else:
                num_in_accum = self.accumulate_gradient
            scale = 1.0 / num_in_accum
        else: scale = 1.0 / self.accumulate_gradient

        # Scale losses for accumulation
        relation_opt, tail_opt = self.optimizers()
        rp_loss = rp_loss_unscaled * scale
        tail_loss = tp_loss_unscaled * scale

        # Backward passes
        self.toggle_optimizer(optimizer=relation_opt, optimizer_idx=0)
        self.manual_backward(rp_loss, retain_graph=(not self.link_head_detached))
        self.untoggle_optimizer(optimizer_idx=0)

        if tail_loss.requires_grad:
            self.toggle_optimizer(optimizer=tail_opt, optimizer_idx=1)
            self.manual_backward(tail_loss, retain_graph=False)
            self.untoggle_optimizer(optimizer_idx=1)

        # Step and zero grads
        is_boundary = ((batch_idx + 1) % self.accumulate_gradient) == 0
        is_last = (total_batches is not None) and ((batch_idx + 1) == total_batches)
        if is_boundary or is_last:
            self.clip_gradients(relation_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            relation_opt.step(); relation_opt.zero_grad()
            self.clip_gradients(tail_opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            tail_opt.step(); tail_opt.zero_grad()

        # Logging unscaled losses
        total_loss = (1.0 - self.loss_weight) * rp_loss_unscaled + self.loss_weight * tp_loss_unscaled
        self.log("train_rp_loss", rp_loss_unscaled, prog_bar=True)
        self.log("train_tp_loss", tp_loss_unscaled, prog_bar=True)
        self.log("train_total_loss", total_loss)
        return {"rp_loss": rp_loss_unscaled, "tp_loss": tp_loss_unscaled}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        logits_rp1, logits_tail1, logits_rp2, logits_tail2 = self.model_forward(batch)
        heads = batch["heads"]
        entity_count_matrix = self.val_head_tail_adjacency[heads.to(self.val_head_tail_adjacency.device)]
        relation_count_matrix = self.val_relation_count_matrix[heads.to(self.val_relation_count_matrix.device)]

        # Losses
        rp_loss, tp_loss = self.compute_phase1_losses(logits_rp1, logits_rp2, logits_tail1, logits_tail2, relation_count_matrix, entity_count_matrix, self.phase1_loss_fn)
        total_loss = (1.0 - self.loss_weight) * rp_loss + self.loss_weight * tp_loss
        
        self.log("valid_rp_loss", rp_loss)
        self.log("valid_tp_loss", tp_loss)
        self.log("valid_total_loss", total_loss, prog_bar=True)

        # Update metrics
        scores_rp, scores_tp = self.generate_scores(logits_rp1, logits_rp2, logits_tail1, logits_tail2)
        self.val_relationMRR.update(heads, scores_rp, relation_count_matrix)
        self.val_relationHitsAt1.update(heads, scores_rp, relation_count_matrix)
        self.val_relationHitsAt3.update(heads, scores_rp, relation_count_matrix)
        self.val_relationHitsAt5.update(heads, scores_rp, relation_count_matrix)
        self.val_relationHitsAt10.update(heads, scores_rp, relation_count_matrix)

        true_tails = self.val_head_tail_adjacency[heads.to(self.val_head_tail_adjacency.device)].to(torch.float32)
        self.val_tailMRR.update(heads, scores_tp, true_tails)
        self.val_tailHitsAt1.update(heads, scores_tp, true_tails)
        self.val_tailHitsAt3.update(heads, scores_tp, true_tails)
        self.val_tailHitsAt5.update(heads, scores_tp, true_tails)
        self.val_tailHitsAt10.update(heads, scores_tp, true_tails)

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        # Log relation metrics (torchmetrics computes and resets them automatically)
        self.log("valid_mrr", self.val_relationMRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_hits1", self.val_relationHitsAt1, on_step=False, on_epoch=True)
        self.log("valid_hits3", self.val_relationHitsAt3, on_step=False, on_epoch=True)
        self.log("valid_hits5", self.val_relationHitsAt5, on_step=False, on_epoch=True)
        self.log("valid_hits10", self.val_relationHitsAt10, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log tail metrics
        self.log("valid_tail_mrr", self.val_tailMRR, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_tail_hits1", self.val_tailHitsAt1, on_step=False, on_epoch=True)
        self.log("valid_tail_hits3", self.val_tailHitsAt3, on_step=False, on_epoch=True)
        self.log("valid_tail_hits5", self.val_tailHitsAt5, on_step=False, on_epoch=True)
        self.log("valid_tail_hits10", self.val_tailHitsAt10, on_step=False, on_epoch=True, prog_bar=True)
        print()

        # Step schedulers if in manual optimization mode
        if self.use_manual_optimization and self.scheduler == "reduce_on_plateau":
            for scheduler in self.lr_schedulers():
                if self.scheduler_monitor in self.trainer.callback_metrics:
                    scheduler.step(self.trainer.callback_metrics[self.scheduler_monitor])
                    new_lr = scheduler.optimizer.param_groups[0]['lr']
                    self.log("lr", new_lr, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        logits_rp1, logits_tail1, logits_rp2, logits_tail2 = self.model_forward(batch)
        heads = batch["heads"]
        relation_count_matrix = self.test_relation_count_matrix[heads.to(self.test_relation_count_matrix.device)]
        entity_count_matrix = self.test_head_tail_adjacency[heads.to(self.test_head_tail_adjacency.device)]

        # Losses
        rp_loss, tp_loss = self.compute_phase1_losses(logits_rp1, logits_rp2, logits_tail1, logits_tail2, relation_count_matrix, entity_count_matrix, self.phase1_loss_fn)
        total_loss = (1.0 - self.loss_weight) * rp_loss + self.loss_weight * tp_loss
        
        self.log("test_rp_loss", rp_loss)
        self.log("test_tp_loss", tp_loss)
        self.log("test_total_loss", total_loss)

        # Update metrics
        scores_rp, scores_tp = self.generate_scores(logits_rp1, logits_rp2, logits_tail1, logits_tail2)
        self.test_relationMRR.update(heads, scores_rp, relation_count_matrix)
        self.test_relationHitsAt1.update(heads, scores_rp, relation_count_matrix)
        self.test_relationHitsAt3.update(heads, scores_rp, relation_count_matrix)
        self.test_relationHitsAt5.update(heads, scores_rp, relation_count_matrix)
        self.test_relationHitsAt10.update(heads, scores_rp, relation_count_matrix)

        true_tails = self.test_head_tail_adjacency[heads.to(self.test_head_tail_adjacency.device)].to(torch.float32)
        self.test_tailMRR.update(heads, scores_tp, true_tails)
        self.test_tailHitsAt1.update(heads, scores_tp, true_tails)
        self.test_tailHitsAt3.update(heads, scores_tp, true_tails)
        self.test_tailHitsAt5.update(heads, scores_tp, true_tails)
        self.test_tailHitsAt10.update(heads, scores_tp, true_tails)

    @torch.no_grad()
    def on_test_epoch_end(self):
        # Log relation metrics
        self.log("test_mrr", self.test_relationMRR, on_step=False, on_epoch=True)
        self.log("test_hits1", self.test_relationHitsAt1, on_step=False, on_epoch=True)
        self.log("test_hits3", self.test_relationHitsAt3, on_step=False, on_epoch=True)
        self.log("test_hits5", self.test_relationHitsAt5, on_step=False, on_epoch=True)
        self.log("test_hits10", self.test_relationHitsAt10, on_step=False, on_epoch=True)
        
        # Log tail metrics
        self.log("test_tail_mrr", self.test_tailMRR, on_step=False, on_epoch=True)
        self.log("test_tail_hits1", self.test_tailHitsAt1, on_step=False, on_epoch=True)
        self.log("test_tail_hits3", self.test_tailHitsAt3, on_step=False, on_epoch=True)
        self.log("test_tail_hits5", self.test_tailHitsAt5, on_step=False, on_epoch=True)
        self.log("test_tail_hits10", self.test_tailHitsAt10, on_step=False, on_epoch=True)

    def generate_scores(self, logits1_rp, logits2_rp, logits1_tp, logits2_tp):
        if self.phase1_loss_fn == 'bce':
            scores_rp = logits1_rp
            scores_tp = logits1_tp
        elif self.phase1_loss_fn == 'poisson':
            scores_rp = logits2_rp
            scores_tp = logits2_tp
        elif self.phase1_loss_fn == 'negative_binomial':
            scores_rp = logits1_rp
            scores_tp = logits1_tp
        elif self.phase1_loss_fn == 'hurdletail':
            prob_non_zero = torch.sigmoid(logits1_tp)
            expected_count = prob_non_zero * torch.exp(logits2_tp)
            scores_rp = logits2_rp
            scores_tp = torch.log(expected_count + 1e-10)
        elif self.phase1_loss_fn == 'hurdlerelation':
            prob_non_zero_rp = torch.sigmoid(logits1_rp)
            expected_count_rp = prob_non_zero_rp * torch.exp(logits2_rp)
            scores_rp = torch.log(expected_count_rp + 1e-10)
            scores_tp = logits2_tp
        elif self.phase1_loss_fn == 'hurdleboth':
            prob_non_zero_rp = torch.sigmoid(logits1_rp)
            expected_count_rp = prob_non_zero_rp * torch.exp(logits2_rp)
            scores_rp = torch.log(expected_count_rp + 1e-10)
            prob_non_zero_tp = torch.sigmoid(logits1_tp)
            expected_count_tp = prob_non_zero_tp * torch.exp(logits2_tp)
            scores_tp = torch.log(expected_count_tp + 1e-10)
        else:
            raise ValueError(f"Unknown phase1_loss_fn: {self.phase1_loss_fn}")
        return scores_rp, scores_tp

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits_rp_bce, logits_tp_bce, logits_rp_poisson, logits_tp_poisson = self.model_forward(batch)
        heads = batch["heads"]
        scores_rp, scores_tp = self.generate_scores(logits_rp_bce, logits_rp_poisson, logits_tp_bce, logits_tp_poisson)
        return {
            "tuples": heads.detach().cpu().unsqueeze(1),  # Use 'tuples' key for compatibility with trainer (contains heads)
            "scores_rp": scores_rp.detach().cpu(),
            "scores_tp": scores_tp.detach().cpu(),
        }