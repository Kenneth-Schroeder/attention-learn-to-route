import gym
from gym import spaces
from problems.op.problem_op import OP
from problems.op.state_op import StateOP
import torch
from utils import move_to
import numpy as np

class OP_env_optimized(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, opts, graph_size):
    super(OP_env_optimized, self).__init__()

    self.opts = opts
    self.num_nodes = graph_size

    obs_dict = {
      'loc': spaces.Box(low=0, high=1, shape=(self.num_nodes, 2)), # remaining node coordinates
      'depot': spaces.Box(low=0, high=1, shape=(2,)), # depot coordinates
      'prize': spaces.Box(low=0, high=np.inf, shape=(self.num_nodes,)), # prizes per node
      'prev_a': spaces.Discrete(self.num_nodes+1), # last action index
      'visited': spaces.MultiBinary(self.num_nodes+1), # visited mask
      'remaining_length': spaces.Box(low=0, high=np.inf, shape=(1,)), # remaining budget
      'action_mask': spaces.MultiBinary(self.num_nodes+1)
    }

    self.observation_space = spaces.Dict(obs_dict)
    self.action_space = spaces.Discrete(self.num_nodes+1)

    self.reset()


  def get_obs(self):
    return {
      'loc': self.coords[1:],
      'depot': self.coords[0],
      'prize': self.prizes_exc_depot,
      'prev_a': self.prev_a,
      'visited': self.visited,
      'remaining_length': self.remaining_length,
      'action_mask': self.forbidden_actions[None, :] # adding a dimension for model
    }


  def step(self, action):
    assert(not self.visited[action].item()), "The node passed to the env's step function was already visited!"
    
    reward = torch.tensor(0)
    if action != 0:
      reward = self.prizes_exc_depot[action-1]

    step_distance = (self.coords[self.prev_a] - self.coords[action]).norm(p=2, dim=-1) # need to fix this to not use loc directly, as depot is not included

    # update prev_a
    if torch.is_tensor(action):
      self.prev_a = action.clone()
    else:
      self.prev_a = torch.tensor(action, device=self.opts.device)

    self.visited[action] = 1
    self.remaining_length -= step_distance

    done = self.prev_a == 0

    dist_to_nodes = (self.coords[action][None, :] - self.coords).norm(p=2, dim=-1)
    potentially_remaining_lengths = self.remaining_length - dist_to_nodes
    # https://stackoverflow.com/questions/3744206/addition-vs-subtraction-in-loss-of-significance-with-floating-points
    self.forbidden_actions = torch.logical_or(self.dist_to_depot > potentially_remaining_lengths, self.visited) # distances from current node to all other nodes, masked by visited nodes, and by remaining length - dist to depot

    info = {} # empty dict
    return self.get_obs(), reward.cpu(), done.cpu(), info


  def reset(self):
    dataset = OP.make_dataset(size=self.num_nodes, num_samples=1, distribution=self.opts.data_distribution)
    depot = dataset.data[0]['depot']
    loc = dataset.data[0]['loc']
    self.coords = move_to(torch.cat((depot[None, :], loc), 0), self.opts.device)

    self.prev_a = torch.tensor(0, device=self.opts.device) # start at depot 0
    self.visited = torch.zeros(self.num_nodes+1, dtype=torch.uint8, device=self.opts.device)

    self.prizes_exc_depot = move_to(dataset.data[0]['prize'], self.opts.device)
    self.remaining_length = move_to(dataset.data[0]['max_length'], self.opts.device)
    self.dist_to_depot = (self.coords[0][None, :] - self.coords).norm(p=2, dim=-1)
    
    self.forbidden_actions = self.dist_to_depot > self.remaining_length - self.dist_to_depot

    return self.get_obs() # reward, done, info can't be included as there are none yet

  def render(self, mode='human'):
    return
  def close (self):
    return