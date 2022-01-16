import gym
from gym import spaces
from problems.tsp.problem_tsp import TSP
from problems.tsp.state_tsp import StateTSP
import torch
from utils import move_to

class TSP_env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, opts):
    super(TSP_env, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    #self.action_space = spaces.Discrete(opts.graph_size)
    # Example for using image as input:
    #self.observation_space = spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self.dataset = TSP.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    self.batch_state = StateTSP.initialize(move_to(torch.stack(self.dataset.data), opts.device))#TSP.make_state(self.dataset.data)
    #self.step(torch.zeros(len(self.dataset.data), dtype=torch.long), torch.device("cuda:0" if opts.use_cuda else "cpu"))

  def step(self, action_batch, device):
    old_batch_state = self.batch_state
    self.batch_state = self.batch_state.update(action_batch)
    rewards = -TSP.get_step_cost(old_batch_state, self.batch_state, device)
    done = self.batch_state.finished[0] # if one is done, all are done in this setting!
    info = None
    return self.batch_state, rewards, done, info

  def reset(self, opts):
    self.dataset = TSP.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    self.batch_state = StateTSP.initialize(move_to(torch.stack(self.dataset.data), opts.device))#TSP.make_state(self.dataset.data)
    return self.batch_state # reward, done, info can't be included as there are none yet

  def render(self, mode='human'):
    return
  def close (self):
    return