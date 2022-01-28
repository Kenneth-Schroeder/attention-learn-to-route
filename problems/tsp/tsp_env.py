import gym
from gym import spaces
from problems.tsp.problem_tsp import TSP
from problems.tsp.state_tsp import StateTSP
import torch
from utils import move_to
import numpy as np

class TSP_env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, opts):
    super(TSP_env, self).__init__()

    self.opts = opts
    num_nodes = opts.graph_size

    obs_dict = {
      'loc': spaces.Box(low=0, high=1, shape=(num_nodes, 2)),
      'dist': spaces.Box(low=0, high=1.415, shape=(num_nodes, num_nodes)),
      'first_a': spaces.Discrete(num_nodes),
      'prev_a': spaces.Discrete(num_nodes),
      'visited': spaces.MultiBinary(num_nodes),
      'length': spaces.Box(low=0, high=np.inf, shape=(1,))
    }

    self.observation_space = spaces.Dict(obs_dict)
    self.action_space = spaces.Discrete(num_nodes)

    self.dataset = TSP.make_dataset(size=self.opts.graph_size, num_samples=1, distribution=self.opts.data_distribution)
    self.batch_state = StateTSP.initialize(move_to(torch.stack(self.dataset.data), self.opts.device))

  def get_obs(self):
    return {
      'loc': self.batch_state.loc.squeeze(),
      'dist': self.batch_state.dist.squeeze(),
      'first_a': self.batch_state.first_a.squeeze(),
      'prev_a': self.batch_state.prev_a.squeeze(),
      'visited': self.batch_state.visited_.squeeze(),
      'length': self.batch_state.lengths.squeeze()
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
    reward = -TSP.get_step_cost(old_batch_state, self.batch_state).item()
    done = self.batch_state.finished.item()
    info = {} # empty dict

    return self.get_obs(), reward, done, info

  def reset(self):
    self.dataset = TSP.make_dataset(size=self.opts.graph_size, num_samples=1, distribution=self.opts.data_distribution)
    self.batch_state = StateTSP.initialize(move_to(torch.stack(self.dataset.data), self.opts.device))

    return self.get_obs() # reward, done, info can't be included as there are none yet

  def render(self, mode='human'):
    return
  def close (self):
    return