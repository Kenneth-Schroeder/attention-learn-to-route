import os
import time
import argparse
import torch
import random
import itertools
import csv
import json
from pathlib import Path


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Attention based model for solving TSP and OP with Reinforcement Learning")

    # Data
    parser.add_argument('--problem', default='tsp', help="The problem to solve, default 'tsp'")
    parser.add_argument('--graph_size', nargs="+", type=int, default=[20], help="The size of the problem graph, if multiple sizes are provided, training and test sets will contain an equal number of problems for each provided size.")
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=5,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--normalization', default='instance', help="Normalization type, 'batch' or 'instance' (default)") # using instance normalization as default
    parser.add_argument('--tanh_clipping', type=float, default=0.0,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')

    # Training
    parser.add_argument('--rl_algorithm', type=str, default='PG', help="Set the RL algorithm to use.")
    parser.add_argument('--lr_actor', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic1', type=float, default=1e-3, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_critic2', type=float, default=1e-3, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_scheduler_type', type=str, default='exp', help='Which decay schedule to use.')
    parser.add_argument('--decay_lr', type=int, default=True, help='Use learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.9999, help='Learning rate decay factor applied after each policy update (after each batch)')

    parser.add_argument('--cyc_base_lr', type=float, default=1e-5, help='Initial learning rate which is the lower boundary of cyclic LR scheduler.')
    parser.add_argument('--cyc_max_lr', type=float, default=1e-4, help='Upper learning rate boundaries in the cyclic LR scheduler')
    parser.add_argument('--cyc_step_size_up', type=int, default=2000, help='Number of training iterations in the increasing half of a cycle.')
    parser.add_argument('--cyc_step_size_down', type=int, default=None, help='Number of training iterations in the decreasing half of a cycle.')

    parser.add_argument('--decay_eps', type=int, default=True, help='Use epsilon-greedy decay')
    
    parser.add_argument('--n_train_envs', type=int, default=32, help='Number of train environments.')
    parser.add_argument('--n_test_envs', type=int, default=64, help='Number of test environments.')

    parser.add_argument('--epc_factor', type=int, default=1, help="'episode_per_collect_factor' - number of episodes to collect from each train env before each network update")
    parser.add_argument('--bs_factor', type=int, default=20, help="'buffer_size_factor' - number batches to fit inside the replay buffer")
    parser.add_argument('--es_factor', type=int, default=100, help="'epoch_size_factor' - number batches to be trained in one epoch")
    parser.add_argument('--te_factor', type=int, default=20, help="'test_episodes_factor' - number of episodes to collect from each test env")
    parser.add_argument('--gamma', type=float, default=1.0, help='Discount Factor')
    parser.add_argument('--n_step', type=int, default=1, help='N-step update length')
    parser.add_argument('--target_freq', type=int, default=100, help='After how many network updates the DQN target network will be updated')
    parser.add_argument('--eps_train', type=float, default=0.5, help='Epsilon-greedy value during training')
    parser.add_argument('--eps_test', type=float, default=0.2, help='Epsilon-greedy value during testing')

    parser.add_argument('--repeat_per_collect', type=int, default=1, help='Number of times policy optimizers learn each batch')
    parser.add_argument('--critic_class_str', type=str, default='v3', help='Which critic architecture to use.')

    parser.add_argument('--eps_clip', type=float, default=0.2, help='PPO Parameter')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='PPO Parameter')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='PPO Parameter')
    parser.add_argument('--gae_lambda', type=float, default=1.00, help='PPO Parameter')

    parser.add_argument('--critics_embedding_dim', type=int, default=64, help='Dimension of input embedding of critics')

    parser.add_argument('--tau', type=float, default=0.005, help='SAC Parameter')
    parser.add_argument('--alpha_ent', type=float, default=None, help='SAC Parameter. Set to None for entropy learning')
    parser.add_argument('--target_ent', type=float, default=-1.0, help='SAC Parameter')
    parser.add_argument('--lr_alpha_ent', type=float, default=3e-4, help='SAC Parameter')

    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=None, help='Random seed to use')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution of the OP per-node rewards, defaults to dist')

    parser.add_argument('--negate_critics_output', type=int, default=True, help='If critics outputs should be negated so let them learn positive values')
    parser.add_argument('--v1critic_activation', type=str, default='leaky', help='Which activation functions to use in critics')
    parser.add_argument('--v1critic_inv_visited', type=int, default=False, help='Whether to pass visited=1 or unvisited=1 values to critics')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Whether to use gradient clipping')
    parser.add_argument('--neg_PG', type=int, default=False, help='Whether to use custom PGPolicy for strictly negative rewards')

    parser.add_argument('--run_name', default='run', help='Name to identify the run')

    parser.add_argument('--args_from_csv', type=str, default=None, help='Extract arguments from csv.')
    parser.add_argument('--csv_row', type=int, default=0, help='Extract arguments from csv at specified row.')

    parser.add_argument('--args_from_json', type=str, default=None, help='Extract arguments from json file.')

    parser.add_argument('--saved_policy_path', type=str, help='Name of saved model.')
    
    parser.add_argument('--gpu_id', default=0, type=int, help='ID of gpu to use.')
    
    opts = parser.parse_args(args)
    gpu_id = opts.gpu_id
    saved_policy_path = opts.saved_policy_path

    def get_opts_from_json(path, graph_size, saved_policy_path=None):
        with open(path) as json_file:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(json_file))
            if saved_policy_path != None:
                t_args.__dict__.update({'saved_policy_path': saved_policy_path})
            return parser.parse_args(namespace=t_args)

    def get_args_from_csv(path, line_number):
        args = []
        with open(path) as f:
            arg_names = next(itertools.islice(csv.reader(f), 0, None)) # uses line 0 for arg names
            arg_values = next(itertools.islice(csv.reader(f), line_number, None)) # which line after header to use as values
            for name, value in zip(arg_names, arg_values):
                if value != '': # will use default value if csv cell empty
                    args.append(f"--{name}")
                    if name == 'graph_size':
                        sizes = value.split()
                        for size in sizes:
                            args.append(size)
                    elif value.lower() in ('yes', 'true', 't', 'y', '1'):
                        args.append("1")
                    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
                        args.append("0")
                    else:
                        args.append(value)
        if saved_policy_path != None:
            args.append("--saved_policy_path")
            args.append(saved_policy_path)
        return args

    epoch_suffix = ''
    if opts.saved_policy_path != None:
        args_stem = Path(saved_policy_path).stem
        if len(split_stem := args_stem.split('_')) > 4:
            args_stem = '_'.join(split_stem[:4])
            epoch_suffix = split_stem[4]
        args_path = f"args/{args_stem}.txt"
        opts = get_opts_from_json(args_path, opts.graph_size, opts.saved_policy_path)
    elif opts.args_from_csv:
        csv_args = get_args_from_csv(opts.args_from_csv, opts.csv_row)
        opts = parser.parse_args(csv_args)
    elif opts.args_from_json: # https://stackoverflow.com/questions/28348117/using-argparse-and-json-together
        opts = get_opts_from_json(opts.args_from_json, opts.graph_size)

    if opts.seed is None:
        opts.seed = random.randint(1,9999)

    opts.use_cuda = torch.cuda.is_available()
    opts.run_name = "{}_{}_{}".format(opts.run_name, epoch_suffix, time.strftime("%Y%m%dT%H%M%S"))
    
    opts.num_graph_sizes = len(opts.graph_size)
    assert opts.n_train_envs % opts.num_graph_sizes == 0, "When providing multiple graph sizes, make sure the number of training envs is divisible by the number of graph sizes"
    assert opts.n_test_envs % opts.num_graph_sizes == 0, "When providing multiple graph sizes, make sure the number of test envs is divisible by the number of graph sizes"
    opts.train_envs_per_size = int(opts.n_train_envs/opts.num_graph_sizes)
    opts.test_envs_per_size = int(opts.n_test_envs/opts.num_graph_sizes)
    
    opts.gpu_id = gpu_id
    
    # save opts
    if not opts.saved_policy_path:
        with open(f"args/{opts.run_name}.txt", 'w') as f:
            f.write(json.dumps(vars(opts)))

    return opts
