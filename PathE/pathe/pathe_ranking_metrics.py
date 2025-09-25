from typing import Optional, Dict

import numpy as np
from torchmetrics import Metric
import torch
from collections import defaultdict

SMALLEST_FLOAT = -torch.finfo(torch.float32).max


class RelationMRRTriples(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict: Dict):
        """
        The constructor
        Parameters
        ----------
        filter_dict : The dictionary mapping between the (h,t) entities and
        the relations between them to filter false negatives
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.all_relations = filter_dict

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------"""

        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 2].item()
            relation = triples[i, 1].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        # CODE ADAPTED FROM PYKEEN ############################################
        # _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 1]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
            # target_relations = target_relations.unsqueeze(1)
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        true_scores = scores[torch.arange(scores.size()[0]), target_relations].unsqueeze(1)
        optimistic_rank = (scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (scores >= true_scores).sum(dim=1)
        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5
        self.reciprocal_ranks += torch.sum(1.0 / realistic_rank, dtype=torch.float)
        self.total += realistic_rank.numel()

        # _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        # target_relations = triples[:, 1]
        # # subtract the target relation ID (index) from all indexes, making
        # # the cell containing the ID (index) of the true relation zero
        # # sorted_indices -= np.expand_dims(relations, 1)
        # target_relations = target_relations.unsqueeze(1)
        # correct_values_to_zero = sorted_indices - target_relations
        # # Get the coordinates of each zero (row, column)
        # zero_indices = (correct_values_to_zero == 0).nonzero()
        # # get the number of the column where each row had a zero and add 1 to
        # # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        # ranks = zero_indices[:, 1] + 1
        # self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        # self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total

class RelationMRRTuples(RelationMRRTriples):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation prediction with tuples (head, relation).
    """
    def __init__(self, filter_dict: Dict):
        """
        The constructor
        Parameters
        ----------
        filter_dict : The dictionary mapping between the (h) entities and its relations to filter false negatives
        """
        super().__init__(filter_dict)
    
    def update(self, triples
               : torch.IntTensor, scores: torch.tensor):
        """
        Update ranking metrics based on model predictions and target relations.

        Parameters
        ----------
        triples : torch.IntTensor
            Tensor of shape (num_samples, 2) containing (head, relation) tuple pairs just named triples instead of tuples for compatibility.
        scores : torch.Tensor
            Tensor of shape (num_samples, num_relations) containing the model's scores
            for each possible relation for each (head, relation) pair.

        Returns
        -------
        None
        """
        tuples = triples

        # Filter out all relations that may be true but are not the one we are predicting
        # This prevents false negatives from affecting the rank of the correct answer
        for i in range(scores.size(0)):
            head = tuples[i, 0].item()
            relation = tuples[i, 1].item()
            for j in self.all_relations[head] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        # Get the target relation indices for each sample
        target_relations = tuples[:, 1]
        # Gather the score assigned by the model to the true relation for each sample
        true_scores = scores[torch.arange(scores.size(0)), target_relations].unsqueeze(1)
        
        # computing ranks

        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        optimistic_rank = (scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (scores >= true_scores).sum(dim=1)

        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        # Update the sum of reciprocal ranks for all samples in the batch
        self.reciprocal_ranks += torch.sum(1.0 / realistic_rank, dtype=torch.float)
        # Update the total number of samples processed
        self.total += realistic_rank.numel()

class RelationMR(Metric):
    """
    A torchmetric implementation of the Mean Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = False

    def __init__(self, filter_dict: Dict):
        """
        The constructor
        Parameters
        ----------
        filter_dict : The dictionary mapping between the (h,t) entities and
        the relations between them to filter false negatives
        """
        super().__init__()
        self.add_state("ranks", default=torch.tensor(0, dtype=
        torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.all_relations = filter_dict


    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 2].item()
            relation = triples[i, 1].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 1]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
        target_relations = target_relations.unsqueeze(1)
        correct_values_to_zero = sorted_indices - target_relations
        # Get the coordinates of each zero (row, column)
        zero_indices = (correct_values_to_zero == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        self.ranks += torch.sum(ranks.float(), dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.ranks / self.total

class RelationHitsAtKTriples(Metric):
    """
    A torchmetric implementation of Hits@K for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict: Dict, k: int):
        """
        The constructor
        Parameters
        ----------
        filter_dict : The dictionary mapping between the (h,t) entities and
        the relations between them to filter false negatives
        k : The rank beyond which the score is ranked at zero
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype= torch.float),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.all_relations = filter_dict
        self.k = k


    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """

        Parameters
        ----------
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------

        """
        for i in range(scores.size()[0]):
            head = triples[i, 0].item()
            tail = triples[i, 2].item()
            relation = triples[i, 1].item()
            # filtering out all relations that may be true but are not the
            # one we are predicting and may increase the rank of the correct
            # answer since they may also be given a high score by the model
            # the -1.0 is arbitrary, may need to be increased to increase
            # the rank of filtered relations depending on the magnitude of
            # the scores
            for j in self.all_relations[(head, tail)] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        # CODE ADAPTED FROM PYKEEN ############################################
        # _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        target_relations = triples[:, 1]
        # subtract the target relation ID (index) from all indexes, making
        # the cell containing the ID (index) of the true relation zero
        # sorted_indices -= np.expand_dims(relations, 1)
            # target_relations = target_relations.unsqueeze(1)
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        true_scores = scores[torch.arange(scores.size()[0]), target_relations].unsqueeze(1)
        optimistic_rank = (scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (scores >= true_scores).sum(dim=1)
        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5
        self.hits += torch.sum((realistic_rank <= self.k), dtype=torch.float)
        self.total += realistic_rank.numel()

        # _, sorted_indices = torch.sort(scores, dim=1, descending=True)
        # target_relations = triples[:, 1]
        # # subtract the target relation ID (index) from all indexes, making
        # # the cell containing the ID (index) of the true relation zero
        # # sorted_indices -= np.expand_dims(relations, 1)
        # target_relations = target_relations.unsqueeze(1)
        # correct_values_to_zero = sorted_indices - target_relations
        # # Get the coordinates of each zero (row, column)
        # zero_indices = (correct_values_to_zero == 0).nonzero()
        # # get the number of the column where each row had a zero and add 1 to
        # # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        # ranks = zero_indices[:, 1] + 1
        # # hits = torch.mean((ranks <= k), dtype=torch.float)
        # self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        # self.total += ranks.numel()

    def compute(self):
        return self.hits / self.total

class RelationHitsAtKTuples(RelationHitsAtKTriples):
    """
    A torchmetric implementation of Hits@K for relation prediction with tuples (head, relation).
    """
    def __init__(self, filter_dict: Dict, k: int):
        super().__init__(filter_dict, k)

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        triples : torch.IntTensor of shape (num_samples, 2) containing (head, relation) tuple pairs. Named triples instead of tuples for compatibility.
        scores : torch.Tensor of shape (num_samples, num_relations)
        """
        tuples = triples

        # Filter out all relations that may be true but are not the one we are predicting
        # This prevents false negatives from affecting the rank of the correct answer
        for i in range(scores.size(0)):
            head = tuples[i, 0].item()
            relation = tuples[i, 1].item()
            for j in self.all_relations[head] - {relation}:
                scores[i, j] = SMALLEST_FLOAT

        target_relations = tuples[:, 1]
        true_scores = scores[torch.arange(scores.size(0)), target_relations].unsqueeze(1)
        optimistic_rank = (scores > true_scores).sum(dim=1) + 1
        pessimistic_rank = (scores >= true_scores).sum(dim=1)
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5
        self.hits += torch.sum((realistic_rank <= self.k), dtype=torch.float)
        self.total += realistic_rank.numel()


class EntityMRRTriples(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True
    # full_state_update = False

    def __init__(self):
        """
        The constructor
        Parameters
        ----------
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of negative head or tail entities
        sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations) or (num-triples, 1) if link prediction

        Returns
        -------
        """
        scores = scores.squeeze()
        triples_per_sample = num_entities_per_sample + 1
        unstacked_triples = triples.reshape(triples.size()[0] // triples_per_sample,
                                            triples_per_sample, -1)

        if scores.ndim > 1:

            scores_unpacked = scores.reshape(triples.size()[0] // triples_per_sample,
                                             triples_per_sample, -1)

            # get the scores of the correct relation for each triple. For each h,r
            # we generate all possible (h,r,t) pairs and thus the score of the
            # (h,r,t)|(h,t) is the score of the head and the correct relation.
            # Thus, the score for each triple is the score produced by the
            # relation predictor for the relation.
            triple_scores = scores_unpacked[torch.arange(scores_unpacked.size()[0]), :,
                            unstacked_triples[:, 0, 1]]  ##### 2 for other
            # format
        else:
            triple_scores = scores.reshape(triples.size()[
                                               0] // triples_per_sample,
                                           triples_per_sample)

        # CODE ADAPTED FROM PYKEEN ############################################
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        true_scores = triple_scores[:,0].unsqueeze(1)
        m = torch.max(triple_scores)
        mi = torch.min(triple_scores)
        optimistic_rank = (triple_scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (triple_scores >= true_scores).sum(dim=1)

        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        self.reciprocal_ranks += torch.sum(1.0 / realistic_rank, dtype=torch.float)
        self.total += realistic_rank.numel()
        #
        # # sort the scores and get the indices
        # _, sorted_indices = torch.sort(triple_scores, dim=1, descending=True)
        #
        # # the correct triple is always the first so looking for index 0
        # zero_indices = (sorted_indices == 0).nonzero()
        # # get the number of the column where each row had a zero and add 1 to
        # # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        # ranks = zero_indices[:, 1] + 1
        #
        # self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        # self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total

class EntityMRRTuples(EntityMRRTriples):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity prediction with tuples.
    """
    def __init__(self):
        raise NotImplementedError("EntityMRRTuples was not tested yet. And is perhaps not even needed.")
        super().__init__()

    def update(self, triples: torch.IntTensor, scores: torch.tensor, num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of negative head or tail entities sampled for each test or val triple
        triples : The tuples the model is being evaluated on, torch.Tensor of shape (num_tuples,2) containing one h,r tuple per row. Named triples for compatibility.
        scores : The scores produced by the model, a torch.Tensor of shape (num_tuples, num_relations) or (num-tuples, 1) if link prediction
        """
        tuples = triples

        scores = scores.squeeze()
        tuples_per_sample = num_entities_per_sample + 1
        unstacked_tuples = tuples.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample, -1)

        # Reshape scores to match tuples
        if scores.ndim > 1:
            scores_unpacked = scores.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample, -1)
            tuple_scores = scores_unpacked[torch.arange(scores_unpacked.size(0)), :, unstacked_tuples[:, 0, 1]]
        else:
            tuple_scores = scores.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample)

        # Compute ranks for the true entity
        true_scores = tuple_scores[:, 0].unsqueeze(1)
        optimistic_rank = (tuple_scores > true_scores).sum(dim=1) + 1
        pessimistic_rank = (tuple_scores >= true_scores).sum(dim=1)
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        self.reciprocal_ranks += torch.sum(1.0 / realistic_rank, dtype=torch.float)
        self.total += realistic_rank.numel()

class CandidateMRRPerSampleFiltered(Metric):
    """
    Filtered MRR per positive sample within candidate groups (group_ids).
    Other positives in the same group are ignored when ranking the current positive.
    """
    higher_is_better = True

    def __init__(self):
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor):
        scores = scores.squeeze()
        labels = labels.to(dtype=torch.float32)
        group_ids = group_ids.to(dtype=torch.long)

        assert scores.ndim == 1
        assert labels.shape == scores.shape
        assert group_ids.shape == scores.shape

        for gid in group_ids.unique():
            group_mask = (group_ids == gid)
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]
            if group_labels.sum() == 0:
                continue

            # Vectorized inner computation: compare all negatives vs all positives
            pos_scores = group_scores[group_labels > 0.5]  # [P]
            neg_scores = group_scores[group_labels < 0.5]  # [N]

            if pos_scores.numel() == 0:
                continue

            if neg_scores.numel() == 0:
                # Only the candidate itself remains -> rank = 1
                realistic_rank = torch.ones_like(pos_scores, dtype=torch.float32, device=pos_scores.device)
            else:
                # gt/ge counts across all negatives for each positive
                gt = (neg_scores.unsqueeze(1) >  pos_scores.unsqueeze(0)).sum(dim=0)       # [P]
                ge = (neg_scores.unsqueeze(1) >= pos_scores.unsqueeze(0)).sum(dim=0)       # [P]
                optimistic_rank = gt + 1          # +1 for the candidate itself
                pessimistic_rank = ge + 1         # includes self in >=
                realistic_rank = 0.5 * (optimistic_rank + pessimistic_rank).to(torch.float32)

            self.reciprocal_ranks += (1.0 / realistic_rank).sum().to(self.reciprocal_ranks.dtype)
            self.total += pos_scores.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total.clamp_min(1)


class EntityHitsAtKTriples(Metric):
    """
    A torchmetric implementation of the hits@K for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, k: int):
        """
        The constructor
        Parameters
        ----------
        k : The rank beyond which the score is ranked at zero
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.k = k

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of negative head or tail entities
        sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations) or (num-triples, 1) if link prediction

        Returns
        -------
        """
        scores = scores.squeeze()

        triples_per_sample = num_entities_per_sample + 1
        unstacked_triples = triples.reshape(triples.size()[
                                                0] // triples_per_sample,
                                            triples_per_sample, -1)

        if scores.ndim > 1:

            scores_unpacked = scores.reshape(triples.size()[
                                                 0] // triples_per_sample,
                                             triples_per_sample, -1)

            # get the scores of the correct relation for each triple. For each h,r
            # we generate all possible (h,r,t) pairs and thus the score of the
            # (h,r,t)|(h,t) is the score of the head and the correct relation.
            # Thus, the score for each triple is the score produced by the
            # relation predictor for the relation.
            triple_scores = scores_unpacked[torch.arange(scores_unpacked.size()[0]), :, unstacked_triples[:, 0, 1]]  ##### 2 for other
            # format
        else:
            triple_scores = scores.reshape(triples.size()[
                                               0] // triples_per_sample,
                                           triples_per_sample)
        # CODE ADAPTED FROM PYKEEN ############################################
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        true_scores = triple_scores[:,0].unsqueeze(1)
        optimistic_rank = (triple_scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (triple_scores >= true_scores).sum(dim=1)

        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        self.hits += torch.sum((realistic_rank <= self.k), dtype=torch.float)
        self.total += realistic_rank.numel()

        # sort the scores and get the indices
        # _, sorted_indices = torch.sort(triple_scores, dim=1, descending=True)
        #
        # # the correct triple is always the first so looking for index 0
        # zero_indices = (sorted_indices == 0).nonzero()
        # # get the number of the column where each row had a zero and add 1 to
        # # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        # ranks = zero_indices[:, 1] + 1
        #
        # self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        # self.total += ranks.numel()

    def compute(self):
        return self.hits / self.total

class EntityHitsAtKTuples(EntityHitsAtKTriples):
    """
    A torchmetric implementation of Hits@K for entity prediction with tuples.
    """
    def __init__(self, k: int):
        raise NotImplementedError("EntityHitsAtKTuples was not tested yet. And is perhaps not even needed.")
        super().__init__(k)

    def update(self, triples: torch.IntTensor, scores: torch.tensor, num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of negative head or tail entities
        sampled
        for each test or val triple
        triples : The tuples the model is being evaluated on, torch.Tensor of shape (num_triples,2) containing one h,r tuple per row, named triples instead of tuples for compatibility.
        scores : The scores produced by the model, a torch.Tensor of shape (num_tuples, num_relations) or (num_tuples, 1) if link prediction
        """
        tuples = triples

        # Remove any singleton dimensions from scores for consistent processing
        scores = scores.squeeze()
        # Each sample consists of the true entity plus num_entities_per_sample negatives
        tuples_per_sample = num_entities_per_sample + 1
        # Group tuples into batches: shape (num_samples, tuples_per_sample, tuple_dim)
        unstacked_tuples = tuples.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample, -1)

        # Reshape scores to match the grouped tuples
        if scores.ndim > 1:
            # Reshape scores to (num_samples, tuples_per_sample, score_dim)
            scores_unpacked = scores.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample, -1)
            # Select the scores for the correct entity index for each tuple
            tuple_scores = scores_unpacked[torch.arange(scores_unpacked.size(0)), :, unstacked_tuples[:, 0, 1]]
        else:
            # If scores are 1D, just reshape to (num_samples, tuples_per_sample)
            tuple_scores = scores.reshape(tuples.size(0) // tuples_per_sample, tuples_per_sample)

        # Compute the score of the true entity for each sample (always at position 0)
        true_scores = tuple_scores[:, 0].unsqueeze(1)
        # Optimistic rank: count how many scores are strictly greater than the true score, then add 1 (1-based rank)
        optimistic_rank = (tuple_scores > true_scores).sum(dim=1) + 1
        # Pessimistic rank: count how many scores are greater or equal to the true score (includes ties)
        pessimistic_rank = (tuple_scores >= true_scores).sum(dim=1)
        # Realistic rank: average of optimistic and pessimistic ranks (handles ties fairly)
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        # Count how many times the true entity is ranked in the top k (Hits@K)
        self.hits += torch.sum((realistic_rank <= self.k), dtype=torch.float)
        # Update the total number of evaluated samples
        self.total += realistic_rank.numel()

class CandidateHitsAtKPerSampleFiltered(Metric):
    """
    Filtered Hits@K per positive sample within candidate groups (group_ids).
    Other positives in the same group are ignored when ranking the current positive.
    """
    higher_is_better = True

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        self.add_state("hits", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor):
        scores = scores.squeeze()
        labels = labels.to(dtype=torch.float32)
        group_ids = group_ids.to(dtype=torch.long)

        assert scores.ndim == 1
        assert labels.shape == scores.shape
        assert group_ids.shape == scores.shape

        for gid in group_ids.unique():
            group_mask = (group_ids == gid)
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]
            if group_labels.sum() == 0:
                continue

            pos_scores = group_scores[group_labels > 0.5]
            neg_scores = group_scores[group_labels < 0.5]

            if pos_scores.numel() == 0:
                continue

            if neg_scores.numel() == 0:
                realistic_rank = torch.ones_like(pos_scores, dtype=torch.float32, device=pos_scores.device)
            else:
                gt = (neg_scores.unsqueeze(1) >  pos_scores.unsqueeze(0)).sum(dim=0)
                ge = (neg_scores.unsqueeze(1) >= pos_scores.unsqueeze(0)).sum(dim=0)
                optimistic_rank = gt + 1
                pessimistic_rank = ge + 1
                realistic_rank = 0.5 * (optimistic_rank + pessimistic_rank).to(torch.float32)

            self.hits += (realistic_rank <= self.k).to(self.hits.dtype).sum()
            self.total += pos_scores.numel()

    def compute(self):
        return self.hits / self.total.clamp_min(1)

class CandidateRecallAtKPerGroup(Metric):
    """
    Recall@K per group (group_ids). For each group:
      recall = (# positives in top-K) / (# positives in group).
    Averaged over groups with at least one positive. No filtering within group.
    """
    higher_is_better = True

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # Optional true positive counts
        self.group_true_counts = None # dict[int, int] or None
    
    def set_true_counts(self, map_group_id_to_true_count: dict):
        """
        Optionally provide the true number of positives per group.
        If provided, this is used instead of counting positives in the labels.
        This is useful if the labels are incomplete (e.g. only a subset of
        positives are in the dataset).
        """
        self.group_true_counts = {int(k): int(v) for k, v in map_group_id_to_true_count.items()}

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor):
        scores = scores.squeeze()
        labels = labels.to(dtype=torch.float32)
        group_ids = group_ids.to(dtype=torch.long)

        assert scores.ndim == 1
        assert labels.shape == scores.shape
        assert group_ids.shape == scores.shape

        for gid in group_ids.unique():
            group_mask = (group_ids == gid)
            group_scores = scores[group_mask]
            group_labels = labels[group_mask]

            k_eff = min(self.k, group_scores.numel())
            topk_idx = torch.topk(group_scores, k=k_eff, largest=True).indices
            num_pos_in_topk = group_labels[topk_idx].sum().item()

            # Denominator: true triple per-group positives if provided; else use positives in dataset
            if self.group_true_counts is not None:
                num_positives = int(self.group_true_counts.get(int(gid.item()), 0))
            else:
                num_positives = int(group_labels.sum().item())

            if num_positives == 0:
                continue    # skip groups with no positives

            self.recall_sum += float(num_pos_in_topk) / float(num_positives)
            self.total += 1

    def compute(self):
        return self.recall_sum / self.total.clamp_min(1)

class CandidateRecallAtKTotal(Metric):
    """
    Recall@K over all samples. 
    recall = (# positives in top-K) / (# positives in dataset).
    No filtering within group.
    """
    higher_is_better = True

    def __init__(self, k: int):
        super().__init__()
        self.k = int(k)
        self.add_state("num_pos_in_topk", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_positives", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def set_num_positives(self, num_positives: int):
        self.total_num_positives = torch.tensor(num_positives)
    
    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor):
        scores = scores.squeeze()

        assert scores.ndim == 1
        assert labels.shape == scores.shape

        k_eff = min(self.k, scores.numel())
        topk_idx = torch.topk(scores, k=k_eff, largest=True).indices
        self.num_pos_in_topk += labels[topk_idx].sum().item()
        self.num_positives += labels.sum().item()

    def compute(self):
        if self.total_num_positives is None or self.total_num_positives == 0:
            return self.num_pos_in_topk / self.total_num_positives.clamp_min(1)
        else:
            return self.num_pos_in_topk / self.num_positives.clamp_min(1)


class EntityMRR_debug(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs in debug mode
    """
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        """
        The constructor
        Parameters
        ----------
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.results_head = []
        self.results_tail = []

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of negative head or tail entities
        sampled for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations) or (num_triples, 1) if doing link prediction

        Returns
        -------
        """

        triples_per_sample = num_entities_per_sample + 1
        unstacked_triples = triples.reshape(triples.size()[
                                                0] // triples_per_sample,
                                            triples_per_sample, -1)

        if scores.ndim > 1:

            scores_unpacked = scores.reshape(triples.size()[
                                                 0] // triples_per_sample,
                                             triples_per_sample, -1)

            # get the scores of the correct relation for each triple. For each h,r
            # we generate all possible (h,r,t) pairs and thus the score of the
            # (h,r,t)|(h,t) is the score of the head and the correct relation.
            # Thus, the score for each triple is the score produced by the
            # relation predictor for the relation.
            triple_scores = scores_unpacked[torch.arange(scores_unpacked.size()[
                                                             0]), :,
                            unstacked_triples[:, 0, 1]]  ##### 2 for other
            # format
        else:
            triple_scores = scores.reshape(triples.size()[
                                               0] // triples_per_sample,
                                           triples_per_sample)
        # CODE ADAPTED FROM PYKEEN ############################################
        # The optimistic rank is the rank when assuming all options with an
        # equal score are placed behind the currently considered. Hence, the
        # rank is the number of options with better scores, plus one, as the
        # rank is one-based.
        true_scores = triple_scores[:,0].unsqueeze(1)
        optimistic_rank = (triple_scores > true_scores).sum(dim=1) + 1

        # The pessimistic rank is the rank when assuming all options with an
        # equal score are placed in front of the currently considered. Hence,
        # the rank is the number of options which have at least the same score
        # minus one (as the currently considered option in included in all
        # options). As the rank is one-based, we have to add 1, which nullifies
        # the "minus 1" from before.
        pessimistic_rank = (triple_scores >= true_scores).sum(dim=1)

        # The realistic rank is the average of the optimistic and pessimistic
        # rank, and hence the expected rank over all permutations of the elements
        # with the same score as the currently considered option.
        realistic_rank = (optimistic_rank + pessimistic_rank).float() * 0.5

        self.reciprocal_ranks += torch.sum(1.0 / realistic_rank, dtype=torch.float)
        self.total += realistic_rank.numel()

        # # sort the scores and get the indices
        # _, sorted_indices = torch.sort(triple_scores, dim=1, descending=True)
        #
        # # the correct triple is always the first so looking for index 0
        # zero_indices = (sorted_indices == 0).nonzero()
        # # get the number of the column where each row had a zero and add 1 to
        # # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        # ranks = zero_indices[:, 1] + 1
        #
        # self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        # self.total += ranks.numel()
        triples_cpu = unstacked_triples.cpu()
        scores_cpu = triple_scores.cpu()
        # sorted_ind_cpu = sorted_indices.cpu()
        ranks_cpu = realistic_rank.cpu()
        for i in range(triples_cpu.size()[0]):
            if triples_cpu[i,1,0] != triples_cpu[i,0,0]:
                head_dump = np.ndarray((2, 4+triples_per_sample))
                head_dump[0,0:3] = triples_cpu[i,0,:].numpy()
                head_dump[0,3] = ranks_cpu[i].numpy()
                heads = triples_cpu[i,:,0]
                # heads_sorted = heads[sorted_ind_cpu[i,:]]
                head_dump[0,4:] = heads.numpy() # heads_sorted if using sort
                head_dump[1,0:3] = triples_cpu[i,0,:].numpy()
                head_dump[1,3] = scores_cpu[i, 0].numpy()
                scores_unsorted = scores_cpu[i,:]
                # scores_sorted = scores_unsorted[sorted_ind_cpu[i,:]]
                head_dump[1,4:] = scores_unsorted.numpy()
                self.results_head.append(head_dump)
            else:
                tail_dump = np.ndarray((2, 4 + triples_per_sample))
                tail_dump[0, 0:3] = triples_cpu[i,0,:].numpy()
                tail_dump[0, 3] = ranks_cpu[i].numpy()
                tails = triples_cpu[i, :, 2]
                # tails_sorted = tails[sorted_ind_cpu[i,:]]
                tail_dump[0, 4:] = tails.numpy() #tails_sorted is using sort
                tail_dump[1, 0:3] = triples_cpu[i,0,:].numpy()
                tail_dump[1, 3] = scores_cpu[i, 0].numpy()
                scores_unsorted = scores_cpu[i, :]
                # scores_sorted = scores_unsorted[sorted_ind_cpu[i,:]]
                tail_dump[1, 4:] = scores_unsorted.numpy()
                self.results_tail.append(tail_dump)


    def compute(self):
        return self.reciprocal_ranks / self.total

    def dump_debug_data(self, dataset_name, negs):
        """
        A function that saves the debug data
         Parameters
        ----------
        dataset_name : The name of the dataset
        negs : The number of negatives
        """
        head_array = np.stack(self.results_head)
        tail_array = np.stack(self.results_tail)
        final = np.stack((head_array, tail_array))
        np.save("../experiments/" + dataset_name + "/" + str(negs) +
                "_debug_train.npy", final)

class AllEntityMRR(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for relation
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """

        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(triple_scores, dim=0, descending=True)

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1

        self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total

class AllEntityHitsAtK(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, k: int):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.k = k

    def update(self, triples: torch.IntTensor, scores: torch.tensor,
               num_entities_per_sample: int):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(triple_scores, dim=0, descending=True)

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 1] + 1
        self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        self.total += 1

    def compute(self):
        return self.hits / self.total

class HeadAllEntityHitsAtK(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, k: int, filter_dict):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.k = k
        self.filter_dict = filter_dict

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        # Filtering###########################
        relation = triples[0, 1].item()
        tail = triples[0, 2].item()
        head = triples[0, 0].item()

        # filter out the actual head and any of the true heads that may exist
        entities_to_filter = self.filter_dict[(relation, tail)] - {head}
        heads_to_filter = triples[:, 0]
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            heads_to_filter_entity = heads_to_filter - entity
            triple_scores[(heads_to_filter_entity == 0)] = SMALLEST_FLOAT
        # select the appropriate heads
        filtered_scores = triple_scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(filtered_scores, dim=0,
                                       descending=True) # FIXME could be
        # wrong simension

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 0] + 1# FIXME weird index only works if one
        # triple at a time
        self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        self.total += 1

    def compute(self):
        return self.hits / self.total

class HeadAllEntityMRR(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.filter_dict = filter_dict

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        # Filtering###########################
        relation = triples[0, 1].item()
        tail = triples[0, 2].item()
        head = triples[0, 0].item()

        # filter out the actual head and any of the true heads that may exist
        entities_to_filter = self.filter_dict[(relation, tail)] - {head}
        heads_to_filter = triples[:, 0]
        zero_indices_list = []
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            heads_to_filter_entity = heads_to_filter - entity
            triple_scores[(heads_to_filter_entity == 0)] = SMALLEST_FLOAT
        # select the appropriate heads
        filtered_scores = triple_scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(filtered_scores, dim=0,
                                       descending=True) # FIXME could be
        # wrong simension

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 0] + 1
        self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        self.total += 1

    def compute(self):
        return self.reciprocal_ranks / self.total

class TailAllEntityHitsAtK(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, k: int, filter_dict):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("hits", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.k = k
        self.filter_dict = filter_dict

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        # Filtering###########################
        relation = triples[0, 1].item()
        head = triples[0, 0].item()
        tail = triples[0, 2].item()

        # filter out the actual head and any of the true heads that may exist
        entities_to_filter = self.filter_dict[(head, relation)] - {tail}
        tails_to_filter = triples[:, 2]  # exclude the true triple from
        # filtering
        # zero_indices_list = []
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            tails_to_filter_entity = tails_to_filter - entity
            triple_scores[(tails_to_filter_entity == 0)] = SMALLEST_FLOAT
            # zero_indices_list.append(
            #     (tails_to_filter_entity == 0).nonzero()[:, 0] + 1) # offset
            # # by one to account for excluding the true triple at index 0
        # zero_indices = torch.cat(zero_indices_list, dim=0)
        # generate a mask with False at the indices to exclude
        # mask = torch.ones(triple_scores.size()[0], dtype=torch.bool)
        # mask[zero_indices] = False
        # select the appropriate heads
        # filtered_scores = triple_scores[mask]
        filtered_scores = triple_scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(filtered_scores, dim=0,
                                       descending=True) # FIXME could be
        # wrong simension

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 0] + 1
        self.hits += torch.sum((ranks <= self.k), dtype=torch.float)
        self.total += 1

    def compute(self):
        return self.hits / self.total

class TailAllEntityMRR(Metric):
    """
    A torchmetric implementation of the Mean Reciprocal Rank for entity
    prediction in Knowledge Graphs
    """
    higher_is_better = True

    def __init__(self, filter_dict):
        """
        The constructor
        Parameters
        ----------
        dataset : The KgLoader object of the Knowledge Graph
        """
        super().__init__()
        self.add_state("reciprocal_ranks", default=torch.tensor(0, dtype=
        torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.filter_dict = filter_dict

    def update(self, triples: torch.IntTensor, scores: torch.tensor):
        """
        Parameters
        ----------
        num_entities_per_sample : The number of head or tail entities sampled
        for each test or val triple
        triples : The triples the model is being evaluated on,
        torch.Tensor of shape (num_triples,3) containing one h,r,t triple
        per row
        scores : The scores produced by the model, a torch.Tensor of shape (
        num_triples, num_relations)

        Returns
        -------
        """
        if scores.ndim > 1:
            triple_scores = scores[torch.arange(scores.size()[0]), triples[0, 1
            ]]
            ##### 2 for other
        else:
            triple_scores = scores

        relation = triples[0, 1].item()
        head = triples[0, 0].item()
        tail = triples[0, 2].item()

        # filter out the actual head and any of the true heads that may exist
        entities_to_filter = self.filter_dict[(head, relation)] - {tail}
        tails_to_filter = triples[:, 2]  # exclude the true triple from
        # filtering
        # get the indices that shoud be excluded
        for entity in entities_to_filter:
            tails_to_filter_entity = tails_to_filter - entity
            triple_scores[(tails_to_filter_entity == 0)] = SMALLEST_FLOAT
        filtered_scores = triple_scores

        # sort the scores and get the indices
        _, sorted_indices = torch.sort(filtered_scores, dim=0,
                                       descending=True) # FIXME could be
        # wrong simension

        # the correct triple is always the first so looking for index 0
        zero_indices = (sorted_indices == 0).nonzero()
        # get the number of the column where each row had a zero and add 1 to
        # convert the zero-indexed index to a rank (0 is 1, 1 is 2 etc.)
        ranks = zero_indices[:, 0] + 1
        self.reciprocal_ranks += torch.sum(1.0 / ranks, dtype=torch.float)
        self.total += ranks.numel()

    def compute(self):
        return self.reciprocal_ranks / self.total
