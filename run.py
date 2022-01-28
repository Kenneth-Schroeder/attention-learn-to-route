#!/usr/bin/env python

import pprint as pp

import torch
import torch.optim as optim

from options import get_options
from nets.attention_model import AttentionModel
from nets.v_estimator import V_Estimator
from utils import load_problem

import tianshou as ts
from problems.tsp.tsp_env import TSP_env
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import ExponentialLR

class Categorical_logits(torch.distributions.categorical.Categorical):
    def __init__(self, logits, validate_args=None):
        super(Categorical_logits, self).__init__(logits=logits, validate_args=validate_args)



def run_DQN(opts):
    problem = load_problem(opts.problem)

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder
    ).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': 1e-3}
    ])

    num_train_envs, num_test_envs = 4, 32
    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])

    gamma, n_step, target_freq = 1.00, 1, 100

    policy = ts.policy.DQNPolicy(actor, optimizer, gamma, n_step, target_update_freq=target_freq)

    epoch, batch_size = 50, 64
    eps_train, eps_test = 0.1, 0.05
    buffer_size = 100000

    num_train_episodes, num_test_episodes = 20, 100 # has to be larger than num_train_env or num_test_env
    step_per_epoch, step_per_collect = 800, 80

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, num_train_episodes), exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)


    result = ts.trainer.offpolicy_trainer( # DOESN'T work with PPO, which makes sense
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        num_test_episodes, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        #logger=logger
    )

def run_PPO(opts):
    problem = load_problem(opts.problem)

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder
    ).to(opts.device)

    critic = V_Estimator(embedding_dim=16).to(opts.device)
    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': 1e-5},
        {'params': critic.parameters(), 'lr': 1e-4}
    ])

    lr_scheduler = ExponentialLR(optimizer, gamma=0.99, verbose=False)

    num_train_envs, num_test_envs = 64, 64
    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])
    gamma = 1.00

    distribution_type = Categorical_logits
    policy = ts.policy.PPOPolicy(actor=actor, 
                                 critic=critic,
                                 optim=optimizer,
                                 dist_fn=distribution_type,
                                 discount_factor=gamma,
                                 lr_scheduler=lr_scheduler,
                                 eps_clip=0.2,
                                 dual_clip=None,
                                 value_clip=False,
                                 advantage_normalization=False,
                                 vf_coef=0.5,
                                 ent_coef=0.01,
                                 gae_lambda=0.95,
                                 reward_normalization=False, # try this
                                 deterministic_eval=False)

    epoch, batch_size = 200, 1024
    buffer_size = 25600

    num_train_episodes, num_test_episodes = 64, 1024 # has to be larger than num_train_env or num_test_env
    step_per_epoch, step_per_collect, repeat_per_collect = 25600, 1280, 2

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, num_train_episodes), exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    writer = SummaryWriter('log_dir')
    logger = TensorboardLogger(writer)

    result = ts.trainer.onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=num_test_episodes,
        batch_size=batch_size,
        step_per_collect=step_per_collect,
        logger=logger
    )

def run_Reinforce(opts):
    problem = load_problem(opts.problem)

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder
    ).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': 1e-4},
    ])

    lr_scheduler = ExponentialLR(optimizer, gamma=0.99, verbose=False)

    num_train_envs, num_test_envs = 4, 32
    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])
    gamma = 1.00

    distribution_type = Categorical_logits
    policy = ts.policy.PGPolicy(model=actor, 
                                optim=optimizer,
                                dist_fn=distribution_type,
                                discount_factor=gamma,
                                lr_scheduler=lr_scheduler,
                                reward_normalization=False,
                                deterministic_eval=False)

    epoch, batch_size = 200, 64
    buffer_size = 5000

    num_train_episodes, num_test_episodes = 20, 100 # has to be larger than num_train_env or num_test_env
    step_per_epoch, step_per_collect, repeat_per_collect = 800, 80, 5

    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, num_train_episodes), exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    writer = SummaryWriter('log_dir')
    logger = TensorboardLogger(writer)

    result = ts.trainer.onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=num_test_episodes,
        batch_size=batch_size,
        step_per_collect=step_per_collect,
        logger=logger
    )


def train(opts):
    # Figure out what's the problem
    #run_DQN(opts)
    #run_Reinforce(opts)
    run_PPO(opts)

    #env = TSP_env(opts)
    #done = False
    #print(env.reset())
    #while(not done):
    #    action = int(input())
    #    print(env.step(action))

    


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    #torch.manual_seed(opts.seed)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    train(opts)


if __name__ == "__main__":
    run(get_options())
