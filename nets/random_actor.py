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

import numpy as np


class Random_Model():
    def solve(self, batch_state: StateTSP, problem, opts):
        
        _new_state = batch_state
        costs = torch.zeros(batch_state.loc.shape[0], device=opts.device) # np.full(batch_state.loc.shape[0], 0)

        batch_actions = torch.stack([
                                torch.randperm(batch_state.loc.shape[1])
                                for _
                                in range(batch_state.loc.shape[0])
                            ]).to(device=opts.device)
        batch_actions = torch.transpose(batch_actions, 0, 1)

        for actions in batch_actions:
            _prev_state = _new_state
            _new_state = _prev_state.update(actions)
            step_costs = problem.get_step_cost(_prev_state, _new_state, opts)
            costs += step_costs.squeeze()

        return costs

    def set_decode_type(self, decode_type, temp=None):
        return

    def eval(self):
        return


