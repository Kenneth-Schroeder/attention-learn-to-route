import os
import time
from tqdm import tqdm
import torch
import math
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from ppo_agent import PPO_Agent, RolloutBuffer



import tianshou as ts
from problems.tsp.tsp_env import TSP_env


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def train_epoch(model, optimizer, opts):

    ppo_agent = PPO_Agent(model, optimizer, opts, discount=1.000, K_epochs=5, eps_clip=0.2, entropy_factor=0.005)
    model.set_decode_type('sampling')
    
    env = TSP_env(opts)
    val_env = TSP_env(opts, num_samples=500)
    max_training_timesteps = 1_000_000
    time_step = 0

    # training loop
    while time_step <= max_training_timesteps:

        state_batch = env.reset(opts)
        #current_ep_reward = 0

        while True:

            # select actions with policy
            actions, log_probs = ppo_agent.select_actions(state_batch)
            ppo_agent.buffer.states.append(state_batch)
            ppo_agent.buffer.actions.append(actions)
            ppo_agent.buffer.logprobs.append(log_probs)

            state_batch, rewards, done, _ = env.step(actions, opts.device)
            ppo_agent.buffer.rewards.append(rewards)
            ppo_agent.buffer.are_done.append(done)

            time_step +=1
            #current_ep_reward += rewards

            # break; if the episode is over
            if done[0]:
                ppo_agent.update()

                if time_step%5 == 0:
                    val_batch = val_env.reset(opts)
                    buffer = RolloutBuffer()
                    while True:
                        actions, _ = ppo_agent.select_actions(val_batch)
                        val_batch, rewards, done, _ = val_env.step(actions, opts.device)
                        buffer.rewards.append(rewards)
                        buffer.are_done.append(done)
                        if done[0]:
                            break
                    _, mean_reward = ppo_agent.validate(buffer)
                    print(f"Step: {time_step}, Mean Reward: {mean_reward}")
                break




def train(model, optimizer, opts):
    num_train_envs, num_test_envs = 20, 32
    train_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_train_envs)])
    test_envs = ts.env.DummyVectorEnv([lambda: TSP_env(opts) for _ in range(num_test_envs)])
    env = TSP_env(opts)

    gamma, n_step, target_freq = 1.00, 1, 100
    policy = ts.policy.DQNPolicy(model, optimizer, gamma, n_step, target_update_freq=target_freq)

    epoch, batch_size = 10, 64
    step_per_epoch, step_per_collect = 1000, 10
    train_num, test_num = 20, 100 # has to be larger than num_train_env or num_test_env
    eps_train, eps_test = 0.1, 0.05
    buffer_size = 100000
    train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=False)
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        #logger=logger
    )

    assert(1==0)


