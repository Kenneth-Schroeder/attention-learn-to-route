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

class V_Estimator(nn.Module):

    def __init__(self,
                 embedding_dim,
                 n_encode_layers=4,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False):
        super(V_Estimator, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        node_dim = 5 # x, y, visited - yes/no, first_a, prev_a - one hot, dist min, max, mean
        # usually node dim is constant w.r.t number of nodes. now if i include distances to all other nodes, that O(n)
        # just take multiple aggregations of the distances, like min, max and mean, which adds 3 dimensions - similar to GNN layers
        
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        # self.graph_embed_to_value = nn.Linear(embedding_dim, 1)
        self.node_embed_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.node_embed_fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.node_embed_to_value = nn.Linear(embedding_dim, 1)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim, # input_dim==embedding_dim as MultiHeadAttentionLayer are used internally
            n_layers=n_encode_layers,
            normalization=normalization
        )



    def forward(self, obs, state=None, info=None):
        loc = obs['loc']
        batch_size, n_loc, _ = loc.shape

        prev_a_idx = obs['prev_a'].view(batch_size, -1)
        first_a_idx = obs['first_a'].view(batch_size, -1)

        prev_a = torch.zeros((batch_size, n_loc, 1), dtype=torch.float)
        first_a = torch.zeros((batch_size, n_loc, 1), dtype=torch.float)

        mask = prev_a_idx.squeeze()!=-1
        replacement_prev_a = torch.nn.functional.one_hot(prev_a_idx[mask], n_loc).view(-1, n_loc, 1).type(torch.float)
        replacement_first_a = torch.nn.functional.one_hot(first_a_idx[mask], n_loc).view(-1, n_loc, 1).type(torch.float)

        prev_a = prev_a.to(device=loc.device)
        first_a = first_a.to(device=loc.device)
        replacement_prev_a = replacement_prev_a.to(device=loc.device)
        replacement_first_a = replacement_first_a.to(device=loc.device)

        prev_a[mask] = replacement_prev_a
        first_a[mask] = replacement_first_a

        visited = obs['visited'].view(batch_size, -1, 1).to(device=loc.device)

        # loc has shape: batch_size, #nodes, #coordinates
        # dist_matrix should have shape: batch_size, #nodes, #nodes
        
        #distances = torch.cdist(loc, loc)
        #distances[distances==0] = torch.median(distances, dim=2).values.flatten()

        #min_distances = torch.min(distances, dim=2).values.view(batch_size, -1, 1)
        #max_distances = torch.max(distances, dim=2).values.view(batch_size, -1, 1)
        #mean_distances = torch.mean(distances, dim=2).view(batch_size, -1, 1)

        # see state_tsp.py ... visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        #action_one_hot = torch.zeros(batch_size, 1, n_loc, dtype=torch.uint8, device=loc.device)
        #action_one_hot = action_one_hot.scatter(-1, batch_act[:, :, None], 1)
        #action_one_hot = action_one_hot.view(batch_size, -1, 1)

        my_input = torch.cat((loc, visited, first_a, prev_a), 2) # , min_distances, max_distances, mean_distances

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(my_input))
        else:
            e = self._init_embed(my_input)
            embeddings, _ = self.embedder(e) # EMBEDDER IS GRAPH ATTENTION ENCODER!

        # graph_embed_mean = embeddings.mean(1)
        # graph_embed_sum = embeddings.sum(1) # TODO use more aggregations and adjust Linear layer.

        embeddings = nn.functional.leaky_relu(self.node_embed_fc1(embeddings), negative_slope=0.2)
        embeddings = nn.functional.leaky_relu(self.node_embed_fc2(embeddings), negative_slope=0.2)
        node_values = self.node_embed_to_value(embeddings).squeeze() # squeeze removes dimensions of size 1
        state_values = -torch.mean(node_values, dim=1)

        return state_values
       

    def _init_embed(self, input):
        # TODO differs for other problems, just focussing on TSP for now
        return self.init_embed(input)