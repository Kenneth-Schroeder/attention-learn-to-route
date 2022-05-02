#!/usr/bin/env python

import pprint as pp

import torch
import torch.optim as optim

from options import get_options
from nets.attention_model import AttentionModel
from nets.v_estimator import V_Estimator
from nets.v_estimator3 import V_Estimator3
from utils import load_problem

import tianshou as ts
from problems.tsp.tsp_env import TSP_env
from problems.tsp.tsp_env_optimized import TSP_env_optimized
#from problems.op.op_env import OP_env
from problems.op.op_env_optimized import OP_env_optimized
from torch.utils.tensorboard import SummaryWriter
from tianshou.data import to_torch, to_torch_as
from tianshou.utils import TensorboardLogger
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR

import numpy as np
from nets.argmaxembed import ArgMaxEmbed
import time
import json

from custom_classes.random import RandomPolicy
from custom_classes.pg import PGPolicy_custom
from custom_classes.discrete_sac import DiscreteSACPolicy_custom

class Categorical_logits(torch.distributions.categorical.Categorical):
    def __init__(self, logits, validate_args=None):
        super(Categorical_logits, self).__init__(logits=logits, validate_args=validate_args)


def updatelog_eps_lr(decay_learning_rate, decay_epsilon, policy, eps, logger, epoch, lr_scheduler=None, env_step=None, batch_size=None, log=False):
    if decay_epsilon:
        policy.set_eps(eps)
    if log:
        logger.write("train/epsilon", epoch, {'Epsilon':eps})
    if lr_scheduler is not None:
        if decay_learning_rate:
            lr_scheduler.step()
        if log:
            logger.write("train/lr", env_step, {'LR':lr_scheduler.get_last_lr()[0]})

def updatelog_lr(decay_learning_rate, logger, lr_schedulers=None, env_step=None, log=False, labels=None):
    if decay_learning_rate and lr_schedulers is not None:
        for scheduler in lr_schedulers:
            scheduler.step()
        if log:
            for scheduler, label  in zip(lr_schedulers, labels):
                logger.write("train/lr", env_step, {label: scheduler.get_last_lr()[0]})



def run_DQN(opts, logger):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6

    learning_rate = opts.lr_actor # 5e-4 
    learning_rate_decay = opts.lr_decay # 0.99995
    lr_scheduler_type = opts.lr_scheduler_type
    decay_learning_rate = opts.decay_lr # False
    decay_epsilon = opts.decay_eps # False
    num_epochs = opts.n_epochs # 100
    num_train_envs = opts.n_train_envs # 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect_factor = opts.epc_factor # 1
    buffer_size_factor = opts.bs_factor # 20
    epoch_size_factor = opts.es_factor # 100
    num_test_envs = opts.n_test_envs # 1024 # has to be smaller or equal to num_test_episodes
    test_episodes_factor = opts.te_factor # 1
    gamma, n_step, target_freq = opts.gamma, opts.n_step, opts.target_freq # 1.00, 1, 100
    eps_train, eps_test = opts.eps_train, opts.eps_test # 0.5, 0.2
    # PER

    num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    episode_per_collect = num_train_envs * episode_per_collect_factor

    batch_size = opts.graph_size * episode_per_collect # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    buffer_size = batch_size * buffer_size_factor

    step_per_epoch = batch_size * epoch_size_factor

    num_test_episodes = num_test_envs * test_episodes_factor # just collect this many episodes using policy and checks the performance
    step_per_collect = batch_size



    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate}
    ])
    lr_scheduler_options = { 
        'exp': ExponentialLR(optimizer, gamma=learning_rate_decay, verbose=False),
        'cyclic': CyclicLR(optimizer, opts.cyc_base_lr, opts.cyc_max_lr, step_size_up=opts.cyc_step_size_up, step_size_down=opts.cyc_step_size_down, mode='triangular2', cycle_momentum=False)
    }
    lr_scheduler = lr_scheduler_options[opts.lr_scheduler_type]

    
    # SubprocVectorEnv DummyVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])

    policy = ts.policy.DQNPolicy(actor, optimizer, gamma, n_step, target_update_freq=target_freq)
    
    

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    result = ts.trainer.offpolicy_trainer( # DOESN'T work with PPO, which makes sense
        policy, train_collector, test_collector, num_epochs, step_per_epoch, step_per_collect,
        num_test_episodes, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: updatelog_eps_lr(decay_learning_rate, decay_epsilon, policy, eps_train/(epoch+1), logger, epoch, lr_scheduler=lr_scheduler, env_step=env_step, batch_size=batch_size, log=True),
        test_fn=lambda epoch, env_step: updatelog_eps_lr(decay_learning_rate, decay_epsilon, policy, eps_test/(epoch+1), logger, epoch, log=False),
        #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger
    )

    torch.save(policy.state_dict(), f"policy_dir/{opts.run_name}.pth")
    #policy.load_state_dict(torch.load("policy.pth"))


def run_PG(opts, logger):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    learning_rate = opts.lr_actor # 5e-5 worked well
    learning_rate_decay = opts.lr_decay # 0.9999
    lr_scheduler_type = opts.lr_scheduler_type
    decay_learning_rate = opts.decay_lr # False
    num_epochs = opts.n_epochs # 100
    num_train_envs = opts.n_train_envs # 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect_factor = opts.epc_factor # 1
    epoch_size_factor = opts.es_factor # 100
    num_test_envs = opts.n_test_envs # 1024 # has to be smaller or equal to num_test_episodes
    test_episodes_factor = opts.te_factor # 1
    gamma = opts.gamma # 1.00
    repeat_per_collect = opts.repeat_per_collect # how many times to learn each batch
    # custom PG loss # skipped, see run from 30th March, ~10:30am
    Policy_class = PGPolicy_custom if opts.neg_PG else ts.policy.PGPolicy


    num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    episode_per_collect = num_train_envs * episode_per_collect_factor

    batch_size = opts.graph_size * episode_per_collect # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    buffer_size = batch_size # no buffer_size_factor for onpolicy algorithms as buffer will be cleared after each network update

    step_per_epoch = batch_size * epoch_size_factor

    num_test_episodes = num_test_envs * test_episodes_factor # just collect this many episodes using policy and checks the performance
    step_per_collect = batch_size



    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate},
    ])
    lr_scheduler_options = { 
        'exp': ExponentialLR(optimizer, gamma=learning_rate_decay, verbose=False),
        'cyclic': CyclicLR(optimizer, opts.cyc_base_lr, opts.cyc_max_lr, step_size_up=opts.cyc_step_size_up, step_size_down=opts.cyc_step_size_down, mode='triangular2', cycle_momentum=False)
    }
    lr_scheduler = lr_scheduler_options[opts.lr_scheduler_type]

    # SubprocVectorEnv DummyVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])

    distribution_type = Categorical_logits
    policy = Policy_class(model=actor,
                          optim=optimizer,
                          dist_fn=distribution_type,
                          discount_factor=gamma,
                          lr_scheduler=lr_scheduler if decay_learning_rate else None, # updates LR each policy update => with each batch 0.9997^(batches_per_epoch*epoch)
                          reward_normalization=False,
                          deterministic_eval=False)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

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
        train_fn=lambda epoch, env_step: logger.write("train/learning_rate", epoch, {'LR':lr_scheduler.get_last_lr()[0]}),
        logger=logger
    )

    torch.save(policy.state_dict(), f"policy_dir/{opts.run_name}.pth")


def run_PPO(opts, logger):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    lr_actor = opts.lr_actor # 1e-4
    lr_critic = opts.lr_critic1 # 1e-4
    learning_rate_decay = opts.lr_decay # 0.9999
    lr_scheduler_type = opts.lr_scheduler_type
    decay_learning_rate = opts.decay_lr # False
    num_epochs = opts.n_epochs # 100
    num_train_envs = opts.n_train_envs # 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect_factor = opts.epc_factor # 1
    epoch_size_factor = opts.es_factor # 100
    num_test_envs = opts.n_test_envs # 1024 # has to be smaller or equal to num_test_episodes
    test_episodes_factor = opts.te_factor # 1
    gamma = opts.gamma # 1.00
    repeat_per_collect = opts.repeat_per_collect # how many times to learn each batch
    critics_embedding_dim = opts.critics_embedding_dim
    eps_clip, vf_coef, ent_coef, gae_lambda = opts.eps_clip, opts.vf_coef, opts.ent_coef, opts.gae_lambda
    critic_class_str = opts.critic_class_str
    critics_class = { 'v1': V_Estimator, 'v3': V_Estimator3 } # critics_class[critic_class_str]



    num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    episode_per_collect = num_train_envs * episode_per_collect_factor

    batch_size = opts.graph_size * episode_per_collect # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    buffer_size = batch_size # no buffer_size_factor for onpolicy algorithms as buffer will be cleared after each network update

    step_per_epoch = batch_size * epoch_size_factor

    num_test_episodes = num_test_envs * test_episodes_factor # just collect this many episodes using policy and checks the performance
    step_per_collect = batch_size





    critic = critics_class[critic_class_str](embedding_dim=critics_embedding_dim, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited, normalization=opts.normalization).to(opts.device)
    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])
    lr_scheduler_options = { 
        'exp': ExponentialLR(optimizer, gamma=learning_rate_decay, verbose=False),
        'cyclic': CyclicLR(optimizer, opts.cyc_base_lr, opts.cyc_max_lr, step_size_up=opts.cyc_step_size_up, step_size_down=opts.cyc_step_size_down, mode='triangular2', cycle_momentum=False)
    }
    lr_scheduler = lr_scheduler_options[opts.lr_scheduler_type]

    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])
    

    distribution_type = Categorical_logits
    policy = ts.policy.PPOPolicy(actor=actor,
                                 critic=critic,
                                 optim=optimizer,
                                 dist_fn=distribution_type,
                                 discount_factor=gamma,
                                 lr_scheduler=lr_scheduler if decay_learning_rate else None,
                                 eps_clip=eps_clip,
                                 dual_clip=None,
                                 value_clip=False,
                                 advantage_normalization=False,
                                 vf_coef=vf_coef,
                                 ent_coef=ent_coef,
                                 gae_lambda=gae_lambda,
                                 reward_normalization=False,
                                 deterministic_eval=False,
                                 max_grad_norm=opts.max_grad_norm)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

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
        logger=logger,
        train_fn=lambda epoch, env_step: logger.write("train/learning_rate", epoch, {'LR':lr_scheduler.get_last_lr()[0]})
    )

    torch.save(policy.state_dict(), f"policy_dir/{opts.run_name}.pth")


def run_SAC(opts, logger):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized } # _optimized

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    lr_actor = opts.lr_actor # 1e-4
    lr_critic1 = opts.lr_critic1 # 1e-5
    lr_critic2 = opts.lr_critic2 # 1e-5
    learning_rate_decay = opts.lr_decay # 0.99995
    lr_scheduler_type = opts.lr_scheduler_type
    decay_learning_rate = opts.decay_lr # False


    num_epochs = opts.n_epochs # 200
    num_train_envs = opts.n_train_envs # 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect_factor = opts.epc_factor # 1
    buffer_size_factor = opts.bs_factor # 1
    epoch_size_factor = opts.es_factor # 100
    num_test_envs = opts.n_test_envs # 1024 # has to be smaller or equal to num_test_episodes
    test_episodes_factor = opts.te_factor # 1
    gamma = opts.gamma # 1.00
    critics_embedding_dim = opts.critics_embedding_dim
    critic_class_str = opts.critic_class_str
    critics_class = { 'v1': V_Estimator, 'v3': V_Estimator3 } # critics_class[critic_class_str]

    tau, alpha = opts.tau, opts.alpha_ent # 0.005, None
    target_ent = opts.target_ent # default = -np.prod(dummy_env.action_space.shape)
    lr_alpha_ent = opts.lr_alpha_ent
    


    num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    episode_per_collect = num_train_envs * episode_per_collect_factor

    batch_size = opts.graph_size * episode_per_collect # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    buffer_size = batch_size * buffer_size_factor

    step_per_epoch = batch_size * epoch_size_factor

    num_test_episodes = num_test_envs * test_episodes_factor # just collect this many episodes using policy and checks the performance
    step_per_collect = batch_size



    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    actor_optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor}
    ])

    critic1 = critics_class[critic_class_str](embedding_dim=critics_embedding_dim, q_outputs=True, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited, normalization=opts.normalization).to(opts.device) # V_Estimator
    critic1_optimizer = optim.Adam([
        {'params': critic1.parameters(), 'lr': lr_critic1}
    ])

    critic2 = critics_class[critic_class_str](embedding_dim=critics_embedding_dim, q_outputs=True, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited, normalization=opts.normalization).to(opts.device)
    critic2_optimizer = optim.Adam([
        {'params': critic2.parameters(), 'lr': lr_critic2}
    ])


    def create_scheduler(optimizer, schedule_type):
        if schedule_type == 'exp':
            return ExponentialLR(optimizer, gamma=learning_rate_decay, verbose=False)
        elif schedule_type == 'cyclic':
            return CyclicLR(optimizer, opts.cyc_base_lr, opts.cyc_max_lr, step_size_up=opts.cyc_step_size_up, step_size_down=opts.cyc_step_size_down, mode='triangular2', cycle_momentum=False)

    lr_scheduler_actor = create_scheduler(actor_optimizer, opts.lr_scheduler_type)
    lr_scheduler_critic1 = create_scheduler(critic1_optimizer, opts.lr_scheduler_type)
    lr_scheduler_critic2 = create_scheduler(critic2_optimizer, opts.lr_scheduler_type)



    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])
    
    if alpha == None:
        target_entropy = target_ent
        log_alpha = torch.zeros(1, requires_grad=True, device=opts.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=lr_alpha_ent)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy_custom(actor=actor, 
                                      actor_optim=actor_optimizer,
                                      critic1=critic1,
                                      critic2=critic2,
                                      critic1_optim=critic1_optimizer,
                                      critic2_optim=critic2_optimizer,
                                      tau=tau,
                                      gamma=gamma,
                                      alpha=alpha,
                                      exploration_noise=None,
                                      reward_normalization=False,
                                      deterministic_eval=False)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, num_epochs, step_per_epoch, step_per_collect,
        num_test_episodes, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: updatelog_lr(decay_learning_rate, logger, lr_schedulers=[lr_scheduler_actor, lr_scheduler_critic1, lr_scheduler_critic2], env_step=env_step, log=True, labels=['ActorLR', 'Critic1LR', 'Critic2LR']),
        logger=logger
    )

    torch.save(policy.state_dict(), f"policy_dir/{opts.run_name}.pth")


def run_A2C(opts, logger):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }

    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    lr_actor = opts.lr_actor # 1e-4
    lr_critic = opts.lr_critic1 # 5e-5
    learning_rate_decay = opts.lr_decay
    lr_scheduler_type = opts.lr_scheduler_type
    decay_learning_rate = opts.decay_lr # False
    num_epochs = opts.n_epochs # 200
    num_train_envs = opts.n_train_envs # 32 # has to be smaller or equal to episode_per_collect
    episode_per_collect_factor = opts.epc_factor # 1
    epoch_size_factor = opts.es_factor # 100
    num_test_envs = opts.n_test_envs # 1024 # has to be smaller or equal to num_test_episodes
    test_episodes_factor = opts.te_factor # 1
    gamma = opts.gamma # 1.00
    repeat_per_collect = opts.repeat_per_collect # how many times to learn each batch
    critics_embedding_dim = opts.critics_embedding_dim # 64
    vf_coef, ent_coef, gae_lambda = opts.vf_coef, opts.ent_coef, opts.gae_lambda
    critic_class_str = opts.critic_class_str
    critics_class = { 'v1': V_Estimator, 'v3': V_Estimator3 } # critics_class[critic_class_str]




    num_of_buffer = num_train_envs # they can't differ, or VectorReplayBuffer will introduce bad data to training
    episode_per_collect = num_train_envs * episode_per_collect_factor

    batch_size = opts.graph_size * episode_per_collect # has to be smaller or equal to buffer_size, defines minibatch size in policy training
    buffer_size = batch_size # no buffer_size_factor for onpolicy algorithms as buffer will be cleared after each network update

    step_per_epoch = batch_size * epoch_size_factor

    num_test_episodes = num_test_envs * test_episodes_factor # just collect this many episodes using policy and checks the performance
    step_per_collect = batch_size



    critic = critics_class[critic_class_str](embedding_dim=critics_embedding_dim, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited, normalization=opts.normalization).to(opts.device)
    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])
    lr_scheduler_options = { 
        'exp': ExponentialLR(optimizer, gamma=learning_rate_decay, verbose=False),
        'cyclic': CyclicLR(optimizer, opts.cyc_base_lr, opts.cyc_max_lr, step_size_up=opts.cyc_step_size_up, step_size_down=opts.cyc_step_size_down, mode='triangular2', cycle_momentum=False)
    }
    lr_scheduler = lr_scheduler_options[opts.lr_scheduler_type]

    # SubprocVectorEnv DummyVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)]) #DummyVectorEnv, SubprocVectorEnv
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])

    distribution_type = Categorical_logits
    policy = ts.policy.A2CPolicy(actor=actor,
                                 critic=critic,
                                 optim=optimizer,
                                 dist_fn=distribution_type,
                                 discount_factor=gamma,
                                 lr_scheduler=lr_scheduler if decay_learning_rate else None,
                                 vf_coef=vf_coef,
                                 ent_coef=ent_coef,
                                 gae_lambda=gae_lambda,
                                 max_grad_norm=opts.max_grad_norm)

    replay_buffer = ts.data.VectorReplayBuffer(total_size=buffer_size, buffer_num=num_of_buffer)
    train_collector = ts.data.Collector(policy, train_envs, replay_buffer, exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

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
        train_fn=lambda epoch, env_step: logger.write("train/learning_rate", epoch, {'LR':lr_scheduler.get_last_lr()[0]}),
        logger=logger
    )

    torch.save(policy.state_dict(), f"policy_dir/{opts.run_name}.pth")


def run_custom_REINFORCE(opts, log_results=False):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }

    critic_class_str = opts.critic_class_str
    critics_class = { 'v1': V_Estimator, 'v3': V_Estimator3 }

    # NOTE: dims must be equivalent to save
    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    critic = critics_class[critic_class_str](embedding_dim=opts.critics_embedding_dim, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited).to(opts.device)

    # https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603/6
    learning_rate = 1e-4
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate}
    ])

    num_test_envs = 512 # has to be smaller or equal to num_test_episodes
    num_train_envs = batch_size = 16
    num_episodes = opts.n_epochs
    num_batches_per_episode = 20
    # DummyVectorEnv, SubprocVectorEnv
    train_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_test_envs)])

    for i in range(num_episodes):
        for j in range(num_batches_per_episode):
            episode_rewards = np.zeros(num_train_envs)
            eposide_log_probs = torch.zeros(num_train_envs, device=opts.device)
            data = ts.data.Batch(obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
            data.obs = train_envs.reset()
            done = False
            embeddings = actor.encode(data.obs)
            while not done:
                logits, _ = actor.decode(data.obs, embeddings)
                if log_results:
                    dist = Categorical_logits(logits)
                    act = dist.sample()
                    eposide_log_probs += dist.log_prob(act)
                data.obs, data.rew, data.done, info = train_envs.step(act)
                episode_rewards += data.rew
                done=data.done[0]
            episode_costs = -to_torch_as(episode_rewards, eposide_log_probs)
            loss = torch.sum(eposide_log_probs * episode_costs)
            print(f"Episode: {i}, b{j}: loss={loss.item()}, rew={np.mean(episode_rewards)}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





def batchify_obs(obs):
    obs['loc'] = torch.unsqueeze(obs['loc'], dim=0)
    obs['dist'] = torch.unsqueeze(obs['dist'], dim=0)
    obs['first_a'] = torch.unsqueeze(obs['first_a'], dim=0)
    obs['prev_a'] = torch.unsqueeze(obs['prev_a'], dim=0)
    obs['visited'] = torch.unsqueeze(obs['visited'], dim=0)
    obs['length'] = torch.unsqueeze(obs['length'], dim=0)
    return obs

def run_STE_argmax(opts):
    problem = load_problem(opts.problem)

    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': opts.lr_actor},
    ])

    env = TSP_env(opts)
    obs = env.reset()
    done = False

    for epoch_idx in range(10):
        epoch_costs = 0
        for _ in range(opts.epoch_size):
            costs = []
            for _ in range(opts.batch_size):
                total_cost = 0
                obs = batchify_obs(obs)
                node_embeddings = model.encode(obs)

                while not done:
                    logits, _ = model.decode(obs, node_embeddings)
                    
                    # create class to make a differentiable argmax operation with embedding selection - done
                    # adjust env to save those embeddings with grads of the whole trajectory - done 
                    # adjust observation to include the right embeddings - done
                    # adjust network to use these embeddings for context creation - done
                    # maybe adjust network to run encoder only once - done
                    # split model into encoder and decoder for easier use here - done
                    # done?

                    action, action_embedding = ArgMaxEmbed.apply(logits, node_embeddings.squeeze()) # NOTE: because of the squeeze here this doesnt work for batches!
                    obs, reward, done, info = env.step(action, action_embedding)
                    obs = batchify_obs(obs)

                    total_cost += reward
                obs, done = env.reset(), False

                costs.append(total_cost)

            # calculate total cost
            costs = torch.tensor(costs, device=opts.device)
            loss = -costs.mean()
            epoch_costs += costs.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(costs.mean())
            #print(loss)
        print(f'Epoch {epoch_idx} Costs: {epoch_costs/opts.epoch_size}') # not working rn

def manual_testing(opts):
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }
    env = problem_env_class[opts.problem](opts)
    obs = env.reset()
    done = False

    print(f"{obs=}")
    while not done:
        obs, reward, done, info = env.step(int(input()))
        print(f"{obs=}, {reward=}, {done=}")


def run_saved(opts, log_solutions=False, logger=None, deterministic_eval=True):
    t0 = time.time()

    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized }
    critic_class_str = opts.critic_class_str
    critics_class = { 'v1': V_Estimator, 'v3': V_Estimator3 }

    # ARCHITECTURE AND PLACEHOLDER ///////////////////////////////////////////////////////////////////////////////////////////////////
    actor = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        output_probs=False,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping
    ).to(opts.device)

    critic1 = critics_class[critic_class_str](embedding_dim=opts.critics_embedding_dim, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited).to(opts.device)
    critic2 = critics_class[critic_class_str](embedding_dim=opts.critics_embedding_dim, problem=problem, negate_outputs=opts.negate_critics_output, activation_str=opts.v1critic_activation, invert_visited=opts.v1critic_inv_visited).to(opts.device)

    # PLACEHOLDERS
    learning_rate = 1e-3 
    optimizer = optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate}
    ])
    critic1_optimizer = optim.Adam([
        {'params': critic1.parameters(), 'lr': learning_rate}
    ])
    critic2_optimizer = optim.Adam([
        {'params': critic2.parameters(), 'lr': learning_rate}
    ])
    gamma, alpha = 1.00, 0.01

    num_eval_envs = 10
    num_runs = 100
    eval_envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_eval_envs)])
    
    # POLICIES /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    distribution_type = Categorical_logits

    if opts.rl_algorithm == 'DQN':
        policy = ts.policy.DQNPolicy(actor, optimizer, gamma, opts.n_step, target_update_freq=opts.target_freq)
    elif opts.rl_algorithm == 'PG':
        Policy_class = PGPolicy_custom if opts.neg_PG else ts.policy.PGPolicy
        policy = Policy_class(model=actor,
                              optim=optimizer,
                              dist_fn=distribution_type,
                              discount_factor=gamma)
    elif opts.rl_algorithm == 'PPO':
        policy = ts.policy.PPOPolicy(actor=actor,
                                     critic=critic1,
                                     optim=optimizer, # NOTE: optimizer originally contains actor and critic params for PPO implementation! but here it is just used as placeholder
                                     dist_fn=distribution_type,
                                     discount_factor=gamma)
    elif opts.rl_algorithm == 'A2C':
        policy = ts.policy.A2CPolicy(actor=actor,
                                     critic=critic1,
                                     optim=optimizer,
                                     dist_fn=distribution_type,
                                     discount_factor=gamma)
    elif opts.rl_algorithm == 'SAC':
        policy = DiscreteSACPolicy_custom(actor=actor, 
                                          actor_optim=optimizer,
                                          critic1=critic1,
                                          critic2=critic2,
                                          critic1_optim=critic1_optimizer,
                                          critic2_optim=critic2_optimizer)
    else:
        print('RL Algorithm specified is not compatible with evaluation mode.')
        return
    
    policy.load_state_dict(torch.load(opts.saved_policy_path)) # f"policy_dir/{opts.save_name}.pth"

    # EVALUATION /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    all_rewards = []
    logs = {}
    logs['runs'] = []

    for i in range(num_runs):
        total_rew = np.zeros(num_eval_envs)
        data = ts.data.Batch(obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
        data.obs = eval_envs.reset()
        not_done_mask = np.ones(num_eval_envs, dtype=bool)

        done = False
        if log_solutions:
            log = {}
            log['coordinates'] = data.obs['loc'].tolist()
            log['tour_probs'] = []
            log['tour_indices'] = []
        embeddings = policy.actor.encode(data.obs) if opts.rl_algorithm != 'DQN' else policy.model.encode(data.obs)
        while not done:
            logits, _ = policy.actor.decode(data.obs, embeddings[not_done_mask]) if opts.rl_algorithm != 'DQN' else policy.model.decode(data.obs, embeddings[not_done_mask])
            dist = Categorical_logits(logits)

            if deterministic_eval:
                act = logits.max(dim=1)[1] # [1] for getting the indices
            else:
                act = dist.sample()

            if log_solutions:
                log['tour_probs'].append(dist.probs.tolist())
                log['tour_indices'].append(act.tolist())

            data.obs, data.rew, data.done, info = eval_envs.step(act, id=np.flatnonzero(not_done_mask))

            total_rew[not_done_mask] += data.rew
            done=np.all(data.done)
            not_done_mask[not_done_mask] = np.logical_not(data.done) # updating all values that were not done before
            data = data[np.logical_not(data.done)]
        all_rewards.append(total_rew)
        if log_solutions:
            logs['runs'].append(log)

    t1 = time.time()
    total_time = t1-t0

    if log_solutions:
        with open(f"eval_logs/{opts.run_name}" + ".json", "w") as fp:
            json.dump(logs, fp)

    all_rewards = np.stack(all_rewards)
    if logger is not None:
        logger.write("eval/rew", opts.graph_size, {'rew': np.mean(all_rewards)})
        logger.write("eval/std", opts.graph_size, {'std': np.std(all_rewards)})
        logger.write("eval/avg_time", opts.graph_size, {'avg_time': total_time/(num_eval_envs*num_runs)})

    print(f"Size: {opts.graph_size}, Mean: {np.mean(all_rewards)}, Std: {np.std(all_rewards)}, Avg Time: {total_time/(num_eval_envs*num_runs)}")





def random_run(opts, logger=None):
    problem = load_problem(opts.problem)
    problem_env_class = { 'tsp': TSP_env_optimized, 'op': OP_env_optimized } # _optimized

    policy = RandomPolicy()

    num_train_envs = 10
    n_episodes = num_train_envs *100

    envs = ts.env.DummyVectorEnv([lambda: problem_env_class[opts.problem](opts) for _ in range(num_train_envs)])
    collector = ts.data.Collector(policy, envs, exploration_noise=False)

    result = collector.collect(n_episode=n_episodes)

    if logger is not None:
        logger.write("eval/rew", opts.graph_size, {'rew': result['rew']})
        logger.write("eval/std", opts.graph_size, {'std': result['rew_std']})

    print(f"Size: {opts.graph_size}, Mean: {result['rew']}, Std: {result['rew_std']}")






def train(opts):
    # python3 run.py --saved_policy_path policy_dir/run_167__20220501T094242.pth
    if opts.saved_policy_path:
        writer = SummaryWriter(f"log_dir/{opts.run_name}")
        writer.add_text("args", str(opts))
        logger = TensorboardLogger(writer, train_interval=1000, test_interval=1, update_interval=1)

        graph_sizes = [20, 30, 40, 50, 100] # [5, 10, 20, 30, 40, 50, 100]
        for graph_size in graph_sizes:
            opts.graph_size = graph_size
            run_saved(opts, logger=logger)
        return
    
    writer = SummaryWriter(f"log_dir/{opts.run_name}")
    writer.add_text("args", str(opts))
    logger = TensorboardLogger(writer, train_interval=1000, test_interval=1, update_interval=1)

    problem_runner = {
        'DQN': run_DQN,
        'PG': run_PG,
        'PPO': run_PPO,
        'SAC': run_SAC,
        'A2C': run_A2C,
        'REINFORCE': run_custom_REINFORCE
    }

    problem_runner[opts.rl_algorithm](opts, logger)

    #run_STE_argmax(opts)
    #manual_testing(opts)
    #random_run(opts)



def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed) # for tianshou random components

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    train(opts)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    run(get_options())
