import gym
from gym import spaces
from problems.op.problem_op import OP
from problems.op.state_op import StateOP
import torch
from utils import move_to
import numpy as np

class OP_env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, opts):
    super(OP_env, self).__init__()

    self.opts = opts
    num_nodes = opts.graph_size

    obs_dict = {
      'loc': spaces.Box(low=0, high=1, shape=(num_nodes, 2)),
      'depot': spaces.Box(low=0, high=1, shape=(2,)),
      'prize': spaces.Box(low=0, high=np.inf, shape=(num_nodes+1,)),
      'prev_a': spaces.Discrete(num_nodes+1),
      'visited': spaces.MultiBinary(num_nodes+1),
      'remaining_length': spaces.Box(low=0, high=np.inf, shape=(1,)),
      'action_mask': spaces.MultiBinary(num_nodes+1)
    }

    self.observation_space = spaces.Dict(obs_dict)
    self.action_space = spaces.Discrete(num_nodes+1)

    self.dataset = OP.make_dataset(size=self.opts.graph_size, num_samples=1, distribution=self.opts.data_distribution)
    data = self.dataset.data[0]
    data.update((k, torch.unsqueeze(move_to(v, self.opts.device), dim=0)) for k,v in data.items())
    self.batch_state = StateOP.initialize(data)

  def get_obs(self):
    return {
      'loc': self.batch_state.coords[:,1:].squeeze(),
      'depot': self.batch_state.coords[:,1].squeeze(),
      'prize': self.batch_state.prize.squeeze()[1:],
      'prev_a': self.batch_state.prev_a.squeeze(),
      'visited': self.batch_state.visited_.squeeze(),
      'remaining_length': self.batch_state.get_remaining_length().squeeze()[None],
      'action_mask': self.batch_state.get_mask()
    }


  def step(self, action):
    old_batch_state = self.batch_state
    
    visited = self.batch_state.visited_.squeeze()
    assert(not visited[action].item()), "The node passed to the env's step function was already visited!"

    masked_action = action
    if visited[masked_action].item():
      non_zero_idxs = (visited == 0).nonzero()
      masked_action = non_zero_idxs[0]

    action_batch = torch.tensor([masked_action], device=self.opts.device)

    self.batch_state = self.batch_state.update(action_batch)
    reward = -OP.get_step_cost(old_batch_state, self.batch_state).item()
    done = self.batch_state.finished.item()
    info = {} # empty dict

    return self.get_obs(), reward, done, info

  def reset(self):
    self.dataset = OP.make_dataset(size=self.opts.graph_size, num_samples=1, distribution=self.opts.data_distribution)
    data = self.dataset.data[0]
    data.update((k, torch.unsqueeze(move_to(v, self.opts.device), dim=0)) for k,v in data.items())
    self.batch_state = StateOP.initialize(data)

    return self.get_obs() # reward, done, info can't be included as there are none yet

  def render(self, mode='human'):
    return
  def close (self):
    return