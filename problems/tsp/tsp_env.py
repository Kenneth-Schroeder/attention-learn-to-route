import gym
from gym import spaces
from utils import torch_load_cpu, load_problem
from problems.tsp.state_tsp import StateTSP
import torch

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
    self.problem = load_problem('tsp')
    self.dataset = self.problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)
    print(torch.stack(self.dataset.data))
    self.batch_state = StateTSP.initialize(torch.stack(self.dataset.data))#TSP.make_state(self.dataset.data)
    print(self.batch_state)
    print(self.batch_state.loc.shape)
    print(self.batch_state.i)
    assert(1==0)

  def step(self, action):
    return #observation, reward, done, info
  def reset(self):
    return #observation  # reward, done, info can't be included
  def render(self, mode='human'):
    return
  def close (self):
    return