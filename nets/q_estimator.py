import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

from problems.tsp.state_tsp import StateTSP
from utils import move_to
from tianshou.policy import BasePolicy
from tianshou.data import Batch

class Q_Estimator(nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_encode_layers=2,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False):
        super(Q_Estimator, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        node_dim = 4 # x, y, in solution, next solution?
        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.graph_embed_to_value = nn.Linear(embedding_dim, 1)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim, # input_dim==embedding_dim as MultiHeadAttentionLayer are used internally
            n_layers=n_encode_layers,
            normalization=normalization
        )

        

    def forward(self, batch: Batch): # batch: tianshou.data.batch.Batch)
        state = batch
        if type(batch) == Batch:
            state = StateTSP.from_experience_batch(batch)

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(state.loc))
        else:
            embeddings, _ = self.embedder(self._init_embed(state.loc)) # EMBEDDER IS GRAPH ATTENTION ENCODER!

        graph_embed_mean = embeddings.mean(1)
        # graph_embed_sum = embeddings.sum(1) # TODO use more aggregations and adjust Linear layer.

        return self.graph_embed_to_value(graph_embed_mean)
       

    def _init_embed(self, input):
        # TODO differs for other problems, just focussing on TSP for now
        return self.init_embed(input)