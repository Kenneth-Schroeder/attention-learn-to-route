#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from options import get_options
from train import train_epoch, train_epoch_sac, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel

from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from utils import torch_load_cpu, load_problem

from tianshou.data import ReplayBuffer
from tianshou.policy import SACPolicy
from nets.q_estimator import Q_Estimator
from nets.v_estimator import V_Estimator


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        assert problem.NAME == 'tsp', "Critic only supported for TSP"
        baseline = CriticBaseline(
            (
                CriticNetworkLSTM(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.tanh_clipping
                )
                if opts.baseline == 'critic_lstm'
                else
                CriticNetwork(
                    2,
                    opts.embedding_dim,
                    opts.hidden_dim,
                    opts.n_encode_layers,
                    opts.normalization
                )
            ).to(opts.device)
        )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizers
    actor_optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    #if 'optimizer' in load_data:
    #    optimizer.load_state_dict(load_data['optimizer'])
    #    for state in optimizer.state.values():
    #        for k, v in state.items():
    #            # if isinstance(v, torch.Tensor):
    #            if torch.is_tensor(v):
    #                state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, lambda epoch: opts.lr_decay ** epoch)



    c1 = Q_Estimator(embedding_dim=16).to(opts.device)
    c2 = Q_Estimator(embedding_dim=16).to(opts.device)
    v = V_Estimator(embedding_dim=16).to(opts.device)

    c1_optimizer = optim.Adam(c1.parameters(), lr=opts.lr_model) # TODO maybe use a different learning rate for critics
    c2_optimizer = optim.Adam(c2.parameters(), lr=opts.lr_model)
    v_optimizer = optim.Adam(v.parameters(), lr=opts.lr_model)


    #:param torch.nn.Module actor: the actor network following the rules in
    #    :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    #:param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    #:param torch.nn.Module critic1: the first critic network. (s, a -> Q(s, a))
    #:param torch.optim.Optimizer critic1_optim: the optimizer for the first
    #    critic network.
    #:param torch.nn.Module critic2: the second critic network. (s, a -> Q(s, a))
    #:param torch.optim.Optimizer critic2_optim: the optimizer for the second
    #    critic network.
    #:param float tau: param for soft update of the target network. Default to 0.005.
    #:param float gamma: discount factor, in [0, 1]. Default to 0.99.
    #:param (float, torch.Tensor, torch.optim.Optimizer) or float alpha: entropy
    #    regularization coefficient. Default to 0.2.
    #    If a tuple (target_entropy, log_alpha, alpha_optim) is provided, then
    #    alpha is automatically tuned.
    #:param bool reward_normalization: normalize the reward to Normal(0, 1).
    #    Default to False.
    #:param BaseNoise exploration_noise: add a noise to action for exploration.
    #    Default to None. This is useful when solving hard-exploration problem.
    #:param bool deterministic_eval: whether to use deterministic action (mean
    #    of Gaussian policy) instead of stochastic action sampled by the policy.
    #    Default to True.
    #:param bool action_scaling: whether to map actions from range [-1, 1] to range
    #    [action_spaces.low, action_spaces.high]. Default to True.
    #:param str action_bound_method: method to bound action to range [-1, 1], can be
    #    either "clip" (for simply clipping the action) or empty string for no bounding.
    #    Default to "clip".
    #:param Optional[gym.Space] action_space: env's action space, mandatory if you want
    #    to use option "action_scaling" or "action_bound_method". Default to None.




    #sac_model = SACPolicy(actor=model, # tianshou.policy.BasePolicy (s -> logits)
    #                      actor_optim=actor_optimizer,
    #                      critic1=c1, # (s, a -> Q(s, a))
    #                      critic1_optim=c1_optimizer,
    #                      critic2=c2, # (s, a -> Q(s, a))
    #                      critic2_optim=c2_optimizer,
    #                      # tau: float = 0.005,
    #                      gamma=1.00, # default float = 0.99,
    #                      # alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2, # entropy coefficient
    #                      # reward_normalization: bool = False,
    #                      # estimation_step: int = 1,
    #                      # exploration_noise: Optional[BaseNoise] = None, # todo check if useful
    #                      # deterministic_eval: bool = True,
    #                      # **kwargs: Any,
    #            )




    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        buffer = ReplayBuffer(size=2048)
        
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            train_epoch_sac(
                model,
                model, # tianshou.policy.BasePolicy (s -> logits)
                actor_optimizer,
                c1, # (s, a -> Q(s, a))
                c1_optimizer,
                c2, # (s, a -> Q(s, a))
                c2_optimizer,
                v,
                v_optimizer,
                # sac_model,
                buffer,
                actor_optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
