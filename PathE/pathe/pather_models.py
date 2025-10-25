"""
PathER : the Path Embedder

This module wraps different architectures for the path embedder, ranging from
self-attention networks inspired by NLP models (BART, GPT, etc.) to denoising
architectures attempting to reconstruct an input sequence or classify the type
of transformation(s) operated on the (discrete) signal.
"""
import math
import logging
from typing import Optional, Any, Union, Callable, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .aggregation import ContextualAggregator, ContextualAggregatorTuples

logger = logging.getLogger(__name__)


class LowRankMultiHead(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = None, bias: bool = True):
        super().__init__()
        if rank is None:
            rank = min(in_features // 2, out_features // 2)  # Automatic rank based on d_model
        self.proj = nn.Linear(in_features, rank, bias=False)   # shared bottleneck
        self.head1 = nn.Linear(rank, out_features, bias=bias)  # first head (e.g., hurdle/log_mu)
        self.head2 = nn.Linear(rank, out_features, bias=bias)  # second head (e.g., count/log_alpha)

    def forward(self, x: torch.Tensor):
        z = F.relu(self.proj(x))
        return self.head1(z), self.head2(z)


def init_transformer_params(module):
    """
    Initialise parameters in a transformer model.
    """
    for p in module.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def init_bert_params(module):
    """
    Initialise the weights specific to the BERT Model.
    This overrides the default init depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).

    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, nn.MultiheadAttention):
        if module._qkv_same_embed_dim:  # torch adpt
            normal_(module.in_proj_weight.data)
        else:  # this is as done in BART (FAIR)
            normal_(module.q_proj_weight.data)
            normal_(module.k_proj_weight.data)
            normal_(module.v_proj_weight.data)


class PositionalEncoding(nn.Module):
    """
    Vanilla Positional Encoding as introduced in the Transformer. Code borrows
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Positional encodings here are implemented as a transformation expecting the
    input sequence to be already embedding before positional info is injected.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1,
                 max_len: int = 5000, batch_first: bool = True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create mockup sequence to serve as additive tensor
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2)
                             * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, 1, embedding_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            A sequence of shape ``[seq_len, bsz, embedding_dim]`` assuming that
            `batch_first = False` else ``[bsz, seq_len, embedding_dim]``
        """
        pos = self.pe[:x.size(0)] if not self.batch_first \
            else torch.swapaxes(self.pe[:x.size(1)], 0, 1)
        x = x + pos

        return self.dropout(x)


class AdaptiveNodeEncoder(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 encodings: torch.tensor,
                 d_output: int,
                 edge_features: torch.tensor = None,
                 dropout: float = 0.1,
                 projector: str = "GCN",
                 activation: Union[str, Callable] = F.leaky_relu) -> None:
        super().__init__()
        self.mlp2 = None
        self.mlp1 = None
        self.projector = projector
        self.d_input = d_output
        self.dropout_p = dropout
        self.incoming_gcn = None
        self.outgoing_gcn = None
        self.activation_f = activation
        self.init_model(encodings, edge_features)

    def init_model(self, encodings, edge_features):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """
        encoding_inc = encodings[1]
        encoding_out = encodings[2]
        self.incoming_gcn = DummyProjector(encodings=encoding_inc,
                                    d_out=self.d_input)
        self.outgoing_gcn = DummyProjector(encodings=encoding_out,
                                    d_out=self.d_input)
        self.mlp1 = nn.Linear(
            self.d_input * 2,
            self.d_input * 2)
        self.mlp2 = nn.Linear(
            self.d_input * 2,
            self.d_input)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape `[num_paths, path length]'

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num paths, path length, embedding dimension]``
            with embeddings

        """
        if x.ndim < 2:
            x.unsqueeze(0)
        inc_encodings = self.incoming_gcn(x)
        out_encodings = self.outgoing_gcn(x)
        logits = self.mlp1(torch.cat((inc_encodings, out_encodings), dim=1))
        logits = self.activation_f(logits)
        logits = self.mlp2(logits)
        embeddings = self.activation_f(logits)
        embeddings = embeddings.reshape((x.size()[0], x.size()[1], -1))
        return embeddings



class DummyProjector(nn.Module):
    """
    A FC layer that performs relation prediction

    Parameters
    ----------
    d_input : int
        The number of expected features in the input as embedding dim.
    output_dim : int
        The number of output neurons (predicted classes - in our case
        relations).
    dim_layers: List[int]
        A list with the dimensions of the hidden layers- default 1 layer of
        512 neurons
    dropout : float
        Dropout value to use in the encoder model (default=0.1).
    activation : Union[str, Callable[[Tensor], Tensor]]
        The activation function of the intermediate layer, can be a string
        ("relu" or "gelu") or a unary callable. Default: relu
    """

    def __init__(self,
                 encodings: torch.tensor,
                 d_out: int,
                 activation: Union[str, Callable] = F.tanh) -> None:
        super().__init__()
        self.model = None
        # self.encodings = encodings
        # Register encodings as a buffer so it follows the module device
        self.register_buffer('encodings', encodings)
        self.activation_f = activation

        self.d_out = d_out
        self.init_model()
        # self.model.apply(self.init_weights)

    def init_model(self):
        """
        A function that initializes the MLP layers
        @return: None
        @rtype: None
        """

        self.model = nn.Sequential(
                                    nn.Dropout(0.5),
                                   nn.Linear(self.encodings.size()[1],
                                             self.d_out))




    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.tensor
            A batch of inputs of shape `[num_paths, path length]'

        Returns
        -------
        logits : torch.tensor
            A tensor of shape ``[num_paths * path length, embedding dim]``

        """
        # self.encodings is a registered buffer and will be on the correct device
        enc = self.encodings[x, :]
        enc_flattened = enc.reshape(enc.size()[0] * enc.size()[1], -1)
        node_features = self.model(enc_flattened)
        # node_features = nn.functional.relu(node_features)
        return node_features



################################################################################
# PathE models
################################################################################

class PathEModelTriples(nn.Module):
    """
    Parameters
    ----------
    vocab_size : int
        The number of token types that can be expected in input sequences
    relcontext_graph : tuple
        ...
    padding_index : int
        The index of the token used to denote padding in the sequences
    d_model : int
        Number of expected features in the encoder/decoder inputs (default=512)
    nhead : int
        Number of heads in the multiheadattention models (default=8)
    num_encoder_layers : int
        Number of sub-encoder-layers in the encoder (default=6)
    dim_feedforward : int
        Dimension of the feedforward network model (default=2048)
    dropout : float
        The dropout value (default=0.1)
    activation: str
        Activation function of encoder/decoder intermediate layer (default=gelu)
    layer_norm_eps : float
        The eps value in layer normalization components (default=1e-5)
    batch_first : bool
        If ``True``, then the input and output tensors are provided
        as (batch, seq, feature); default=True (batch, seq, feature)
    norm_first : bool
        If ``True``, encoder and decoder layers will perform LayerNorms before
        other attention and feedforward ops, otherwise after (default=false)
    max_seqlen : int
        If provided, it will allocate ``max_seqlen`` positional encodings for
        Entity-Relation paths, and ``math.ceil(max_seqlen / 2)`` for R paths.
    """

    def __init__(self,
                 vocab_size: int,
                 relcontext_graph: tuple,
                 padding_idx: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 ent_aggregation : str = "avg",
                 num_agg_heads : int = 1,
                 num_agg_layers : int = 1,
                 laf_units : int = 1,
                 context_heads : int = 1,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 node_projector: str = "GCN",
                 max_seqlen: int = None,
                ) -> None:

        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.batch_first = batch_first
        assert node_projector in ["dummy"]
        self.node_projector = node_projector
        self.ent_aggregation = ent_aggregation
        
        if self.ent_aggregation == "avg":
            logger.info("Using average aggregation for entities")
        else:  # assume contextual aggregation
            self.aggregator = ContextualAggregator(
                embedding_dim=d_model, aggregator=self.ent_aggregation,
                num_agg_heads=num_agg_heads, num_agg_layers=num_agg_layers,
                laf_units=laf_units, context_heads=context_heads,
            )

        max_seqlen = vocab_size if max_seqlen is None else max_seqlen
        # Relational Embedding layers are shared between encoder and decoder;
        # in contrast to vanilla Parth, positional embeddings are not shared.
        self.rel_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.pred_link_embeddings = nn.Embedding(
            num_embeddings=vocab_size - 2,
            embedding_dim=d_model,
        )
        self.rel_index_tensor = nn.Parameter(
            torch.arange(vocab_size - 2), requires_grad=False)
        self.pos_embeddings = nn.Embedding(
            num_embeddings=max_seqlen,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Transformer layers: encoder and decoder needed for representations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            activation, layer_norm_eps, batch_first, norm_first)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)
        # Relational context project for init entities
        self.rcontext_projector = AdaptiveNodeEncoder(
            encodings=relcontext_graph, edge_features=None,
            d_output=d_model, dropout=dropout, projector=self.node_projector,
            # activation=  # TODO custom act
        )
        # if self.ent_aggregation == "mha":
        #     self.entity_fusion_head = torch.nn.MultiheadAttention(
        #         d_model, 2, batch_first=True)  # FIXME
        #     self.head_query_vector = nn.Embedding(1, d_model)
        #     self.entity_fusion_tail = torch.nn.MultiheadAttention(
        #         d_model, 2, batch_first=True)  # FIXME
        #     self.tail_query_vector = nn.Embedding(1, d_model)

        # Relation prediction head from averaged entities
        # self.relpredict_head_avg1 = nn.Linear(d_model * 2, d_model)  # FIXME
        # self.relpredict_head_avg2 = nn.Linear(d_model, vocab_size-2)
        #
        self.relpredict_head_avg = nn.Linear(d_model * 2, vocab_size - 2)


        ###################################################
        # Relation-based agrgeagtion and link prediction 
        # self.relpredict_head_l1 = nn.Parameter(torch.rand((
        #     vocab_size - 2, d_model * 2), dtype=torch.float32))
        #
        # self.gating_layer = nn.Parameter(torch.rand((
        #     vocab_size - 2, d_model * 2), dtype=torch.float32))
        #
        self.link_predict_head1 = nn.Linear(d_model * 3, d_model)
        self.link_predict_head2 = nn.Linear(d_model, 1)
        # XXX Apply custom initialisation
        init_transformer_params(self.encoder)
        # self.apply(init_bert_params)

    def device(self):
        return self.rel_embeddings.weight.device

    def link_prediction(self, heads, tails, relation_scores):
        relation_embeddings = self.pred_link_embeddings(
            self.rel_index_tensor).unsqueeze(0)
        relation_weights = nn.functional.softmax(relation_scores)
        relations_emb = torch.matmul(relation_weights,
                                     relation_embeddings).squeeze()
        heads_emb, tails_emb = self.mha_aggregate(heads, tails, relations_emb)
        triples = torch.cat((heads_emb, relations_emb, tails_emb), dim=1)
        logits = nn.functional.relu(self.link_predict_head1(triples))
        logits = self.link_predict_head2(logits)
        return logits

    def mha_aggregate(self, heads, tails, relations):
        expanded_relation_emb = relations.unsqueeze(1)
        heads_aggregated, _ = self.head_aggregator(
            query=expanded_relation_emb, key=heads,
            value=heads, need_weights=False)
        tails_aggregated, _ = self.tail_aggregator(
            query=expanded_relation_emb, key=tails,
            value=tails, need_weights=False)
        return heads_aggregated.squeeze(), tails_aggregated.squeeze()

    def average_aggregate(self, heads, tails):
        unique_heads, counts_h = torch.unique(heads.nonzero()[:, 0],
                                              return_counts=True)
        unique_tails, count_tails = torch.unique(tails.nonzero()[:, 0],
                                                 return_counts=True)
        count_h, count_t = (counts_h.float() / heads.size()[2],
                            count_tails.float() / tails.size()[2])
        heads_emb = torch.sum(heads, dim=1) / count_h.unsqueeze(1)
        tails_emb = torch.sum(tails, dim=1) / count_t.unsqueeze(1)
        return heads_emb, tails_emb

    def predict_relations_and_average(self, head_emb, tail_embed, ppt):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.relpredict_head_avg1(head_tail)
        predictions = nn.functional.relu(predictions)
        predictions = self.relpredict_head_avg2(predictions)
        triple_preds = torch.split(predictions, ppt.tolist(), dim=0)
        padded_triples = nn.utils.rnn.pad_sequence(triple_preds,
                                                  batch_first=True)
        # unique_triples, counts = torch.unique(padded_triples.nonzero()[:, 0],
        #                                       return_counts=True)
        # counts = counts.float() / padded_triples.size()[2]
        averaged_scores = torch.sum(padded_triples, dim=1) / ppt.unsqueeze(1)
        return averaged_scores

    def predict_links_and_average(self, head_emb, tail_embed, ppt):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.linkpred_head_avg1(head_tail)
        predictions = nn.functional.relu(predictions)
        predictions = self.linkpred_head_avg2(predictions)
        triple_preds = torch.split(predictions, ppt.tolist(), dim=0)
        padded_triples = nn.utils.rnn.pad_sequence(triple_preds,
                                                   batch_first=True)
        # unique_triples, counts = torch.unique(padded_triples.nonzero()[:, 0],
        #                                       return_counts=True)
        # counts = counts.float() / padded_triples.size()[2]
        averaged_scores = torch.sum(padded_triples, dim=1) / ppt.unsqueeze(1)
        return averaged_scores

    def average_embeddings(self, embeddings, ppt, idxs):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """
        # hack so that the only indexes that are negative are the non-existent
        idxs_local= idxs + 1
        # split the idxs for each triple (numtriple, numpath)
        split_idxs = torch.split(idxs_local, ppt.tolist(), dim=0)
        padded_idxs = nn.utils.rnn.pad_sequence(split_idxs, batch_first=True)
        # get the positive indexes (true ones)
        actual_idxs = (padded_idxs > 0).nonzero()
        # get their number per triple
        unique, counts = torch.unique(actual_idxs[:,0], return_counts=True)

        # split the embeddings based on the existing number of entities per
        # triple
        triple_embeds = torch.split(embeddings, counts.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(triple_embeds,
                                                   batch_first=True)
        # average ignoring padding
        averaged_embs = torch.sum(padded_embs, dim=1) / counts.unsqueeze(1)
        return averaged_embs

    def sum_embeddings(self, embeddings, ppt, idxs):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """
        # hack so that the only indexes that are negative are the non-existent
        idxs_local= idxs + 1
        # split the idxs for each triple (numtriple, numpath)
        split_idxs = torch.split(idxs_local, ppt.tolist(), dim=0)
        padded_idxs = nn.utils.rnn.pad_sequence(split_idxs, batch_first=True)
        # get the positive indexes (true ones)
        actual_idxs = (padded_idxs > 0).nonzero()
        # get their number per triple
        unique, counts = torch.unique(actual_idxs[:,0], return_counts=True)

        # split the embeddings based on the existing number of entities per
        # triple
        triple_embeds = torch.split(embeddings, counts.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(triple_embeds,
                                                   batch_first=True)
        # average ignoring padding
        summed_embs = torch.sum(padded_embs, dim=1)
        return summed_embs

    def predict_relation_from_ht(self, head_emb, tail_embed):
        head_tail = torch.cat((head_emb, tail_embed), dim=1)
        predictions = self.relpredict_head_avg(head_tail)
        # predictions = self.relpredict_head_avg1(head_tail)
        # predictions = nn.functional.relu(predictions)
        # predictions = self.relpredict_head_avg2(predictions)
        return predictions

    def link_predict_from_ht(self, head_emb, tail_embed, targets):
        relations_emb = self.pred_link_embeddings(targets)
        triples = torch.cat((head_emb, relations_emb, tail_embed), dim=1)
        predictions = self.link_predict_head1(triples)
        predictions = nn.functional.relu(predictions)
        predictions = self.link_predict_head2(predictions)
        return predictions

    def interleave_embeddings(self, ent_embeddings, rel_embeddings):
        # print(f"Rel embed {rel_embed.shape}\nEnt embed {ent_embed.shape}")
        # Interleave rels and ents and injecting positional information
        # path_embed = torch.stack((ent_embed, rel_embed), dim=2).view(2, 6, 5)
        path_embed = torch.zeros((ent_embeddings.size()[0],  # bsz
                                  ent_embeddings.size()[1] + rel_embeddings.size()[1],  # np
                                  ent_embeddings.size()[2]), device=self.device())

        ent_ind = torch.arange(0, ent_embeddings.size()[1] + rel_embeddings.size()[1],
                               2, device=self.device())
        rel_ind = torch.arange(1, ent_embeddings.size()[1] + rel_embeddings.size()[1],
                               2, device=self.device())
        path_embed[:, ent_ind, :], path_embed[:, rel_ind, :] = \
            ent_embeddings, rel_embeddings
        return path_embed

    def select_separated_head_and_tail_embeddings(self,memory, head_idxs,
                                                  tail_idxs, entity_origin):
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # print(f"Memory {memory.shape}")
        # head and tail selection with variable num paths for each
        head_emb = memory[torch.arange(memory.shape[0]), head_idxs, :]
        tail_emb = memory[torch.arange(memory.shape[0]), tail_idxs, :]
        # when the number of tail and head embeddings is not the same
        # and the entity_origin is available
        head_idxs_mask = ~entity_origin.bool()
        # head_indices = head_idxs[head_idxs_mask]
        tail_idxs_mask = ~head_idxs_mask
        # tail_indices = tail_idxs[tail_idxs_mask]
        head_embed = head_emb[head_idxs_mask, :]
        tail_embed = tail_emb[tail_idxs_mask, :]
        return head_embed, tail_embed


    def select_head_and_tail_embeddings(self, memory, head_idxs,
                                                  tail_idxs):
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # print(f"Memory {memory.shape}")
        # head and tail selection with variable num paths for each
        head_emb = memory[torch.arange(memory.shape[0]), head_idxs, :]
        tail_embed = memory[torch.arange(memory.shape[0]), tail_idxs, :]
        # when the number of tail and head embeddings is not the same
        head_indices = (head_idxs >= 0).nonzero().squeeze()
        head_embed = head_emb[head_indices, :]
        tail_indices = (tail_idxs >= 0).nonzero().squeeze()
        tail_embed = tail_embed[tail_indices, :]
        return head_embed, tail_embed

    def average_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        # average ignoring padding
        averaged_embs = torch.sum(padded_embs, dim=1) / paths_per_entity.unsqueeze(1)
        return averaged_embs

    def split_and_pad_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        return padded_embs
    
    def split_and_pad_embeddings_with_origin_and_entities(self, embeddings, origin, target, entities):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """
        raise NotImplementedError("This function may not work correctly and should not be needed since the ppt function exists.")
        output, inverse, counts = torch.unique_consecutive(entities, return_inverse=True, return_counts=True)
        # calculate paths per entity based on target origin
        start_indices = torch.cumsum(counts, dim=0) - counts
        origin_grouped = origin[start_indices]
        count_selection_mask = (origin_grouped == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        return padded_embs
    
    def split_and_pad_embeddings_with_ppt(self, embeddings, ppt):
        """
        Splits and pads entity embeddings according to the number of paths per tuple or triple.

        Given a tensor of entity embeddings and a tensor specifying the number of paths per tuple/triple (ppt),
        this function splits the embeddings accordingly and pads them to create a uniform batch.

        embeddings : torch.Tensor
            Tensor of shape (num_entities, dim) containing the entity embeddings.
        ppt : torch.Tensor
            1D tensor of shape (num_tuples_or_triples,) specifying the number of paths per tuple/triple.

        Returns
        -------
        torch.Tensor
            A padded tensor of shape (num_tuples_or_triples, max_num_paths, dim), where max_num_paths is the maximum
            in ppt. Shorter sequences are padded with zeros.
        """
        # split the embeddings based on the existing number of paths per tuple
        entity_embeds = torch.split(embeddings, ppt.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        return padded_embs

    def sum_embeddings_with_origin(self, embeddings, origin, target):
        """
        A function that averages the embeddings in a table for each triple
        Parameters
        ----------
        embeddings : The emebeddings of shape (num_entities, dim)
        ppt : The paths per triple
        idxs : The idxs of the entities in each path

        Returns
        -------
        A single embedding table of shape (num triples, dim)
        """

        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == target)
        paths_per_entity = counts[count_selection_mask]
        # split the embeddings based on the existing number of paths per entity
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        # average ignoring padding
        # summed_embs = torch.sum(padded_embs, dim=1)
        summed_embs, _ = torch.max(padded_embs, dim=1)
        return summed_embs

    def query_head_embeddings(self, embeddings, origin):
        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == 0)
        paths_per_entity = counts[count_selection_mask]
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        query_vector = self.head_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            padded_embs.size()[0],-1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=padded_embs,
            value=padded_embs, need_weights=False)
        return result.squeeze()

    def query_tail_embeddings(self, embeddings, origin):
        """
        A function that aggregates the tail embeddings by using a specialized
         tail query vector and multi head attention
        Parameters
        ----------
        embeddings : the embeddings to aggregate (num paths, embeddings
        dimension)
        origin : the vector denoting which entity head - 0  or tail - 1,
        each embedding belongs to (num paths, 1)

        Returns
        -------
        A single vector for each tail of each triple

        """
        output, counts = torch.unique_consecutive(origin, return_counts=True)
        count_selection_mask = (output == 1)
        paths_per_entity = counts[count_selection_mask]
        entity_embeds = torch.split(embeddings, paths_per_entity.tolist(), dim=0)
        # convert list tp tensor by padding
        padded_embs = nn.utils.rnn.pad_sequence(entity_embeds, batch_first=True)
        query_vector = self.tail_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            padded_embs.size()[0],-1, -1)
        result, _ = self.entity_fusion_tail(
            query=expanded_query, key=padded_embs,
            value=padded_embs, need_weights=False)
        return result.squeeze()

    def embed_nodes(self, ent_paths: Tensor,
                    rel_paths: Tensor,
                    idxs: Tensor,
                    pos: Tensor,
                    ppe: Optional[Tensor] = None,
                    source_mask: Optional[Tensor] = None
                    ):
        assert self.batch_first == True, "Dev. code in batch-first mode"
        # FIXME remove after testing
        # injector = EmbeddingInjector(torch.arange(1,100,
        #                                                dtype=torch.int32,
        #                                           device=self.device()),
        #                                   64, 14505, self.device())
        rel_embed = self.rel_embeddings(rel_paths)
        ent_embed = self.rcontext_projector(ent_paths)
        # test = injector.forward(ent_embed, ent_paths) # FIXME remove after
        #  testing
        # interleave embeddings to create path embeddings
        path_embed = self.interleave_embeddings(ent_embeddings=ent_embed,
                                                rel_embeddings=rel_embed)
        path_embed += self.pos_embeddings(pos)
        # Masking and feeding the entity-relation path to the encoder
        # print(f"Input {path_embed.shape}")
        source_pmask = None  # FIXME
        memory = self.encoder(path_embed,
                              mask=source_mask,
                              src_key_padding_mask=source_pmask)
        # Isolate the embeddings for head and tails only from the triples
        # head_embed is (num_paths, d_model) after squeezing
        # head and tail selection with variable num paths for each
        ent_emb = memory[torch.arange(memory.shape[0]), idxs, :]

        if self.ent_aggregation != "avg":

            entity_embeds = torch.split(ent_emb, ppe.tolist(),
                                        dim=0)
            # padded_embs = nn.utils.rnn.pad_sequence(entity_embeds,
            #                                         batch_first=True)

            return entity_embeds
        else:  # use the average pooling otherwise
            # split the embeddings based on the existing number of paths per entity
            entity_embeds = torch.split(ent_emb, ppe.tolist(),
                                        dim=0)
            # convert list tp tensor by padding
            padded_embs = nn.utils.rnn.pad_sequence(entity_embeds,
                                                    batch_first=True)
            # average ignoring padding
            averaged_embs = torch.sum(padded_embs,
                                      dim=1) / ppe.unsqueeze(1) #FIXME summation
            # averaged_embs = torch.sum(padded_embs, dim=1)
            # averaged_embs, _ = torch.max(padded_embs, dim=1)
            return averaged_embs

    def mha_aggregate_heads(self, head_emb):
        query_vector = self.head_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            head_emb.size()[0], -1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=head_emb,
            value=head_emb, need_weights=False)
        return result.squeeze()

    def mha_aggregate_tails(self,tail_emb):
        query_vector = self.tail_query_vector(
            torch.tensor([0], device=self.device()))
        expanded_query = query_vector.unsqueeze(0).expand(
            tail_emb.size()[0], -1, -1)
        result, _ = self.entity_fusion_head(
            query=expanded_query, key=tail_emb,
            value=tail_emb, need_weights=False)
        return result.squeeze()



    def predict_links(self, head_emb, tail_emb, targets):
        if self.ent_aggregation != "avg":
            head_emb, tail_emb = self.aggregator(
            head_emb, tail_emb)
        logits = self.link_predict_from_ht(head_emb, tail_emb, targets)
        return logits

    def predict_relations(self, head_emb, tail_emb):
        if self.ent_aggregation != "avg":
            head_emb, tail_emb = self.aggregator(
            head_emb, tail_emb)
        logits = self.predict_relation_from_ht(head_emb, tail_emb)
        logits = nn.functional.softmax(logits, dim=1)
        return logits


    def forward(self,
                ent_paths: Tensor,
                rel_paths: Tensor,
                head_idxs: Tensor,
                tail_idxs: Tensor,
                pos: Tensor,
                entity_origin: Tensor,
                targets: Tensor,
                ppt: Optional[Tensor] = None,
                source_mask: Optional[Tensor] = None):

        assert self.batch_first == True, "Dev. code in batch-first mode"

        rel_embed = self.rel_embeddings(rel_paths)
        ent_embed = self.rcontext_projector(ent_paths)
        # interleave embeddings to create path embeddings
        path_embed = self.interleave_embeddings(ent_embeddings=ent_embed,
                                                rel_embeddings=rel_embed)
        path_embed += self.pos_embeddings(pos)
        # Masking and feeding the entity-relation path to the encoder
        # print(f"Input {path_embed.shape}")
        source_pmask = None  # FIXME
        memory = self.encoder(path_embed,
                              mask=source_mask,
                              src_key_padding_mask=source_pmask)
        # would be needed for grouping without entity_origin
        # entity_idxs = torch.where(head_idxs >= 0, head_idxs, tail_idxs)
        # heads = ent_paths[torch.arange(ent_paths.size(0)), entity_idxs//2]

        # Isolate the embeddings for head and tails only from the triples
        head_embed, tail_embed = (
            self.select_separated_head_and_tail_embeddings(memory=memory,
                                                           head_idxs=head_idxs,
                                                           tail_idxs=tail_idxs,
                                                           entity_origin=entity_origin))
        if self.ent_aggregation != "avg":
            head_padded = self.split_and_pad_embeddings_with_origin(
                head_embed, origin=entity_origin, target=0)
            tail_padded = self.split_and_pad_embeddings_with_origin(
                tail_embed, origin=entity_origin, target=1)
            head_emb, tail_emb = self.aggregator(
                head_padded, tail_padded)
        else:  # use the average pooling otherwise
            tail_emb = self.average_embeddings_with_origin(
                tail_embed, origin=entity_origin, target=1)
            head_emb = self.average_embeddings_with_origin(
                head_embed, origin=entity_origin, target=0)

        head_emb = head_emb.unsqueeze(0) if head_emb.ndim < 2 else head_emb
        tail_emb = tail_emb.unsqueeze(0) if tail_emb.ndim < 2 else tail_emb

        logits_rp = self.predict_relation_from_ht(head_emb, tail_emb)

        # When entities from validation/test splits are not in the training set, they may have no paths.
        # This causes the aggregated embedding tensors (head_emb, tail_emb) to be smaller than the
        # batch size (targets.size(0)), leading to size mismatches.
        # The following logic reconstructs full-size embedding tensors, padding with zeros for entities
        # that had no paths.
        if head_emb.size(0) != targets.size(0) or tail_emb.size(0) != targets.size(0):
            logger.warning(f"Head/tail embeddings size mismatch with targets size ({head_emb.size(0)}, {tail_emb.size(0)} vs {targets.size(0)}). Reconstructing full-size tensors with zeros for missing paths.")
            
            # Determine which triples in the batch actually have head and tail paths
            split_origins = torch.split(entity_origin, ppt.tolist())
            has_head_paths = torch.tensor([torch.any(o == 0) for o in split_origins], device=self.device())
            has_tail_paths = torch.tensor([torch.any(o == 1) for o in split_origins], device=self.device())

            # Create full-size tensors padded with zeros
            head_emb_full = torch.zeros(targets.size(0), self.d_model, device=self.device(), dtype=head_emb.dtype)
            if head_emb.numel() > 0:
                head_emb_full[has_head_paths] = head_emb
            
            tail_emb_full = torch.zeros(targets.size(0), self.d_model, device=self.device(), dtype=tail_emb.dtype if tail_emb.numel() > 0 else head_emb.dtype)
            if tail_emb.numel() > 0:
                tail_emb_full[has_tail_paths] = tail_emb
            
            head_emb = head_emb_full
            tail_emb = tail_emb_full

        logits_link = self.link_predict_from_ht(head_emb, tail_emb, targets)

        return logits_rp, logits_link

class PathEModelTuples(PathEModelTriples):
    def __init__(self,
                 vocab_size: int,
                 relcontext_graph: tuple,
                 padding_idx: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 ent_aggregation : str = "avg",
                 num_agg_heads : int = 1,
                 num_agg_layers : int = 1,
                 laf_units : int = 1,
                 context_heads : int = 1,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 node_projector: str = "GCN",
                 max_seqlen: int = None,
                ) -> None:
        
        super().__init__(vocab_size, relcontext_graph, padding_idx, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, 
                         ent_aggregation, num_agg_heads, num_agg_layers, laf_units, context_heads, batch_first, norm_first, node_projector, max_seqlen,
                )
        # correct aggregator class perhaps copy entire super init instead of fixing afterwards
        if self.ent_aggregation == "avg":
            # check if this works with tuples instead of triples
            logger.info("Using average aggregation for entities")
        else:  # assume contextual aggregation
            self.aggregator = ContextualAggregatorTuples(
                embedding_dim=self.d_model, aggregator=self.ent_aggregation,
                num_agg_heads=num_agg_heads, num_agg_layers=num_agg_layers,
                laf_units=laf_units, context_heads=context_heads,
            )
        
        #adjust dimension because only heads and not heads and tails are used
        self.relation_head = nn.Linear(d_model, vocab_size - 2)
        self.link_predict_head1 = nn.Linear(d_model * 2, d_model)
        self.link_predict_head2 = nn.Linear(d_model, 1)

        # Tail classification head over all entities (predict t | h)
        # Infer number of entities from the relational context encodings (offset=2)
        try:
            num_entities = int(relcontext_graph[0].shape[0] - 2)
        except Exception:
            # Fallback in case the provided relcontext_graph has unexpected structure
            num_entities = vocab_size - 2  # safe fallback; will be corrected by trainer usage
        self.num_entities = num_entities
        self.tail_head = nn.Linear(d_model, self.num_entities)
    
    def select_separated_head_embeddings(self, memory, head_idxs, entity_origin):
        """
        Selects and returns the head entity embeddings from the memory tensor for tuples.

        Parameters
        ----------
        memory : torch.Tensor
            The contextualized memory tensor output from the encoder, of shape (batch_size, seq_len, d_model).
        head_idxs : torch.Tensor
            Tensor of indices indicating the positions of head entities in the sequence, of shape (batch_size,).
        entity_origin : torch.Tensor
            Tensor indicating the origin of each entity (e.g., 0 for head, 1 for tail), of shape (batch_size,).

        Returns
        -------
        torch.Tensor
            The selected head entity embeddings, filtered by entity_origin, of shape (num_heads, d_model).
        """
        # Isolate the embeddings for heads only from the tuples
        # head_embed is (num_paths, d_model) after squeezing
        # print(f"Memory {memory.shape}")
        # head and tail selection with variable num paths for each
        head_emb = memory[torch.arange(memory.shape[0]), head_idxs, :]
        # when the number of tail and head embeddings is not the same
        # and the entity_origin is available
        head_idxs_mask = ~entity_origin.bool()
        # head_indices = head_idxs[head_idxs_mask]
        head_embed = head_emb[head_idxs_mask, :]
        return head_embed

# currently not usable because rel_idxs do not exist
    def select_relation_embeddings(self, memory, rel_idxs):
        # Selects relation embeddings from memory using rel_idxs
        rel_emb = memory[torch.arange(memory.shape[0]), rel_idxs, :]
        return rel_emb

    def forward(self,
                ent_paths: Tensor,
                rel_paths: Tensor,
                head_idxs: Tensor,
                pos: Tensor,
                entity_origin: Tensor,
                ppt: Optional[Tensor] = None,
                source_mask: Optional[Tensor] = None,
                detach_link_head: bool = True):

        assert self.batch_first == True, "Dev. code in batch-first mode"

        # Get static relation embeddings for each relation in the batch
        rel_embed = self.rel_embeddings(rel_paths)
        # Get contextualized entity embeddings for each entity path
        ent_embed = self.rcontext_projector(ent_paths)
        # Interleave entity and relation embeddings to form the path representation
        path_embed = self.interleave_embeddings(ent_embeddings=ent_embed,
                                                rel_embeddings=rel_embed)
        path_embed += self.pos_embeddings(pos)
        # Masking and feeding the entity-relation path to the encoder
        # Pass through the transformer encoder to get contextualized memory
        source_pmask = None  # No padding mask used here
        memory = self.encoder(path_embed,
                              mask=source_mask,
                              src_key_padding_mask=source_pmask)
        # Select only the head embeddings from the contextualized memory
        head_embed = self.select_separated_head_embeddings(memory=memory, head_idxs=head_idxs, entity_origin=entity_origin)

        heads = ent_paths[torch.arange(ent_paths.size(0)), head_idxs//2]
        # Select relation embeddings from the contextualized memory (if needed)
        # not working because of missing rel_idxs
        # relation_embed = self.select_relation_embeddings(memory=memory, rel_idxs=rel_paths)

        # Aggregate head embeddings (no tails in tuple mode)
        if self.ent_aggregation != "avg":
            # print(f"head_idxs {head_idxs.shape} {head_idxs[:5]} \nent_paths {ent_paths.shape} {ent_paths[:15]}\n pos {pos.shape} {pos[:5]}\n entity_origin {entity_origin.shape} {entity_origin[:5]}")
            head_padded = self.split_and_pad_embeddings_with_ppt(head_embed, ppt)
            head_emb = self.aggregator(head_padded)
        else:  # use the average pooling otherwise
            raise NotImplementedError(
                "Average aggregation for tuples is not implemented yet. Needs function using ppe like in 'avg' case.")
            # tail_emb = self.average_embeddings_with_origin(
            #     tail_embed, origin=entity_origin, target=1)
            head_emb = self.average_embeddings_with_origin(
                head_embed, origin=entity_origin, target=0)

        # Ensure head_emb has the correct shape
        head_emb = head_emb.unsqueeze(0) if head_emb.ndim < 2 else head_emb

        # Relation prediction P(r | h) (always with gradients)
        logits_rp1, logits_rp2 = self.predict_relation_from_h(head_emb)
        
        # Tail prediction P(t | h, r) (optionally detached)
        logits_tail1, logits_tail2 = self.tail_predict_from_h(head_emb.detach() if detach_link_head else head_emb)


        return logits_rp1, logits_tail1, logits_rp2, logits_tail2

    def predict_relation_from_h(self, head_emb):
        return self.relation_head(head_emb), None

    def link_predict_from_h(self, head_emb, targets):
        # Get relation embeddings for the target relations
        relations_emb = self.pred_link_embeddings(targets)
        # Concatenate head and relation embeddings (no tail embedding)
        tuples = torch.cat((head_emb, relations_emb), dim=1)
        # Pass through link prediction head
        predictions = self.link_predict_head1(tuples)
        predictions = nn.functional.relu(predictions)
        predictions = self.link_predict_head2(predictions)
        return predictions
    
    def tail_predict_from_h(self, head_emb: torch.Tensor):
        return self.tail_head(head_emb), None

    # ---------------------------
    # DIFFERENCE TO TRIPLES PREDICTION:
    # - In triples prediction, the model uses (head, relation, tail) and concatenates all three embeddings for link prediction.
    # - Here, only (head, relation) tuples are used: tail embeddings are not selected, aggregated, or concatenated.
    # - The prediction heads (linear layers) are adjusted to accept only the concatenated head and relation embeddings (dimension d_model*2 instead of d_model*3).
    # - All aggregation and selection logic is simplified to ignore tails.
    # - This setup is suitable for tasks like predicting if a (head, relation) tuple is valid or for relation classification given only the head.


class PathEModelTuplesMultiHead(PathEModelTuples):
    def __init__(self,
                 vocab_size: int,
                 relcontext_graph: tuple,
                 padding_idx: int,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 ent_aggregation : str = "avg",
                 num_agg_heads : int = 1,
                 num_agg_layers : int = 1,
                 laf_units : int = 1,
                 context_heads : int = 1,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 node_projector: str = "GCN",
                 max_seqlen: int = None,
                 relation_multi_head: bool = True,
                 tail_multi_head: bool = True,
                ) -> None:
        
        super().__init__(vocab_size, relcontext_graph, padding_idx, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, activation, layer_norm_eps, 
                         ent_aggregation, num_agg_heads, num_agg_layers, laf_units, context_heads, batch_first, norm_first, node_projector, max_seqlen,
                )
        # override the heads to be LowRankMultiHead
        if relation_multi_head:
            self.rel_heads = LowRankMultiHead(d_model, vocab_size - 2)
        else:
            self.rel_heads = nn.Linear(d_model, vocab_size - 2)
        if tail_multi_head:
            self.tail_heads = LowRankMultiHead(d_model, self.num_entities)
        else:
            self.tail_heads = nn.Linear(d_model, self.num_entities)
    
    def predict_relation_from_h(self, head_emb):
        if isinstance(self.rel_heads, LowRankMultiHead):
            return self.rel_heads(head_emb)
        else:
            return self.rel_heads(head_emb), None
    
    def tail_predict_from_h(self, head_emb: torch.Tensor):
        if isinstance(self.tail_heads, LowRankMultiHead):
            return self.tail_heads(head_emb)
        else:
            return self.tail_heads(head_emb), None
