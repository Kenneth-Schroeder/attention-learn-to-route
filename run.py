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

    num_epochs = 200
    
    num_train_envs = 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect = num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    buffer_size = opts.graph_size * episode_per_collect
    batch_size = buffer_size # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    repeat_per_collect = 1 
    step_per_epoch = buffer_size * 100

    num_test_envs = 1024 # has to be smaller or equal to num_test_episodes
    num_test_episodes = 1024 # just collect this many episodes using policy and checks the performance
    step_per_collect = buffer_size

    num_train_envs, num_test_envs = 4, 32
    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])

    gamma, n_step, target_freq = 1.00, 1, 100
    policy = ts.policy.DQNPolicy(actor, optimizer, gamma, n_step, target_update_freq=target_freq)
    eps_train, eps_test = 0.1, 0.05

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    result = ts.trainer.offpolicy_trainer( # DOESN'T work with PPO, which makes sense
        policy, train_collector, test_collector, num_epochs, step_per_epoch, step_per_collect,
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
        {'params': actor.parameters(), 'lr': 1e-4},
        {'params': critic.parameters(), 'lr': 1e-4}
    ])

    lr_scheduler = ExponentialLR(optimizer, gamma=0.99, verbose=False)

    num_epochs = 200
    
    num_train_envs = 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect = num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    buffer_size = opts.graph_size * episode_per_collect
    batch_size = buffer_size # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    repeat_per_collect = 1 
    step_per_epoch = buffer_size * 100

    num_test_envs = 1024 # has to be smaller or equal to num_test_episodes
    num_test_episodes = 1024 # just collect this many episodes using policy and checks the performance

    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])
    gamma = 1.00

    distribution_type = Categorical_logits
    policy = ts.policy.PGPolicy(model=actor, 
                                optim=optimizer,
                                dist_fn=distribution_type,
                                discount_factor=gamma,
                                #lr_scheduler=lr_scheduler,
                                reward_normalization=False,
                                deterministic_eval=False)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

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
                                 gae_lambda=1.00,
                                 reward_normalization=False,
                                 deterministic_eval=False)


    writer = SummaryWriter('log_dir')
    logger = TensorboardLogger(writer)

    result = ts.trainer.onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=num_test_episodes,
        batch_size=batch_size,
        episode_per_collect=episode_per_collect,
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

    num_epochs = 200
    
    num_train_envs = 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect = num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    buffer_size = opts.graph_size * episode_per_collect
    batch_size = buffer_size # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    repeat_per_collect = 1 
    step_per_epoch = buffer_size * 100

    num_test_envs = 1024 # has to be smaller or equal to num_test_episodes
    num_test_episodes = 1024 # just collect this many episodes using policy and checks the performance

    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])
    gamma = 1.00

    distribution_type = Categorical_logits
    policy = ts.policy.PGPolicy(model=actor, 
                                optim=optimizer,
                                dist_fn=distribution_type,
                                discount_factor=gamma,
                                #lr_scheduler=lr_scheduler,
                                reward_normalization=False,
                                deterministic_eval=False)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    writer = SummaryWriter('log_dir')
    logger = TensorboardLogger(writer)

    result = ts.trainer.onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=num_epochs,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=repeat_per_collect,
        episode_per_test=num_test_episodes,
        batch_size=batch_size,
        episode_per_collect=episode_per_collect,
        logger=logger
    )


def train(opts):
    # Figure out what's the problem
    run_DQN(opts)
    #run_Reinforce(opts)
    #run_PPO(opts)



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
