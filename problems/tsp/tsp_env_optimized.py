import gym
from gym import spaces
from problems.tsp.problem_tsp import TSP
from problems.tsp.state_tsp import StateTSP
import torch
from utils import move_to
import numpy as np


# just a single TSP problem
class TSP_env_optimized(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, opts, graph_size):
    super(TSP_env_optimized, self).__init__()

    self.opts = opts
    self.num_nodes = graph_size

    obs_dict = {
      'loc': spaces.Box(low=0, high=1, shape=(self.num_nodes, 2)),
      #'dist': spaces.Box(low=0, high=1.415, shape=(num_nodes, num_nodes)),
      'first_a': spaces.Discrete(self.num_nodes),
      'prev_a': spaces.Discrete(self.num_nodes),
      'visited': spaces.MultiBinary(self.num_nodes),
      #'length': spaces.Box(low=0, high=np.inf, shape=(1,)),
      'action_mask': spaces.MultiBinary(self.num_nodes)
    }

    self.observation_space = spaces.Dict(obs_dict)
    self.action_space = spaces.Discrete(self.num_nodes)

    self.reset()


  def get_obs(self):
    return {
      'loc': self.loc,
      'first_a': self.first_a,
      'prev_a': self.prev_a,
      'visited': self.visited,
      'action_mask': (self.visited > 0)[None, :] # adding a dimension for model, more complicated mask for OP
    }


  def step(self, action):
    assert(not self.visited[action].item()), "The node passed to the env's step function was already visited!"
    # optional action masking, need to remove assert for this to work
    #masked_action = action
    #if self.visited[masked_action].item():
    #  non_zero_idxs = (visited == 0).nonzero()
    #  masked_action = non_zero_idxs[0]

    # calculate step length
    if self.first_a == -1:
      cost = torch.tensor(0, device=self.opts.device)
    else:
      cost = (self.loc[self.prev_a] - self.loc[action]).norm(p=2, dim=-1)

    # update prev_a, first_a, visited and num_visited
    if torch.is_tensor(action):
      self.prev_a = action.clone()
    else:
      self.prev_a = torch.tensor(action, device=self.opts.device)

    if self.first_a == -1:
      self.first_a = self.prev_a.clone()
    self.visited[action] = 1
    self.num_visited += 1

    done = self.num_visited >= self.num_nodes
    if done:
      cost += (self.loc[self.prev_a] - self.loc[self.first_a]).norm(p=2, dim=-1)

    info = {} # empty dict
    return self.get_obs(), -cost.cpu(), done, info

  def reset(self):
    dataset = TSP.make_dataset(size=self.num_nodes, num_samples=1, distribution=self.opts.data_distribution)
    self.loc = move_to(dataset.data[0], self.opts.device)
    self.prev_a = torch.tensor(-1, device=self.opts.device)
    self.first_a = torch.tensor(-1, device=self.opts.device)
    self.visited = torch.zeros(self.num_nodes, dtype=torch.uint8, device=self.opts.device)
    self.num_visited = 0

    return self.get_obs() # reward, done, info can't be included as there are none yet

  def render(self, mode='human'):
    return
  def close (self):
    return