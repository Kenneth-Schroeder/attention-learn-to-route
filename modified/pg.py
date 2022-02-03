from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_torch
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd

from modified.base import BasePolicy_custom


class PGPolicyTraj(BasePolicy_custom):
    """Implementation of REINFORCE algorithm.
    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        lr_scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.
        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i
        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.
        :return: A :class:`~tianshou.data.Batch` which has 4 keys:
            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        logits, hidden = self.actor(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act = logits.argmax(-1)
            elif self.action_type == "continuous":
                act = logits[0]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        num_episodes = np.count_nonzero(batch.done)
        episode_len = int(len(batch.done)/num_episodes)
        for _ in range(repeat):
            episode_rewards = []
            episode_log_probs = []
            all_rewards = []
            all_log_probs = []
            for episode_idx in range(num_episodes):
                start_idx = episode_idx*episode_len
                episode = batch[start_idx:start_idx+episode_len]


                result = self(episode)



                dist = result.dist
                act = to_torch_as(episode.act, result.act)


                #print("//////////")
                #print(batch.logits)
                #print(act)
                #print(result.logits)
                #assert(1==0)

                #print(batch.logits.gather(dim=1, index=act[:, None]))
                
                #my_dist = self.dist_fn(episode.logits)
                my_log_probs = episode.logits.gather(dim=1, index=act[:, None]) #my_dist.log_prob(act)
                #print(dist.log_prob(act))
                #assert(1==0)


                rewards = to_torch(episode.rew, device=result.act.device)
                #episode_reward = rewards.sum()
                #episode_rewards.append(episode_reward)
                #log_prob = my_log_probs
                ##log_prob = dist.log_prob(act)
                #episode_log_prob = log_prob.sum()
                #episode_log_probs.append(episode_log_prob)

                all_rewards.append(rewards)
                all_log_probs.append(my_log_probs)

            #episode_rewards = torch.stack(episode_rewards)
            #episode_log_probs = torch.stack(episode_log_probs)
            episode_rewards = torch.stack(all_rewards).sum(dim=1)
            episode_log_probs = torch.stack(all_log_probs).sum(dim=1)

            print(episode_rewards)
            print(episode_log_probs)
            loss = -(episode_rewards * episode_log_probs).mean()

            self.optim.zero_grad()
            loss.backward(retain_graph=True)
            self.optim.step()
            losses.append(loss.item())

        # update learning rate if lr_scheduler is given
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {"loss": losses}
