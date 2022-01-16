import gym
from gym import spaces
from problems.tsp.problem_tsp import TSP
from problems.tsp.state_tsp import StateTSP
import torch
import torch.optim as optim

from nets.v_estimator import V_Estimator

class RolloutBuffer:
	def __init__(self):
		self.actions = []
		self.states = []
		self.logprobs = []
		self.rewards = []
		self.are_done = []
		#('state', 'action', 'log_prob', 'next_state', 'reward', 'reward_to_come')

	def clear(self):
		del self.actions[:]
		del self.states[:]
		del self.logprobs[:]
		del self.rewards[:]
		del self.are_done[:]

class PPO_Agent():
	def __init__(self, model, optimizer, opts, discount, K_epochs, eps_clip):
		self.policy = model
		self.optimizer = optimizer
		self.critic = V_Estimator(embedding_dim=16).to(opts.device)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=opts.lr_critic)
		self.device = opts.device

		self.discount = discount
		self.eps_clip = eps_clip
		self.K_epochs = K_epochs

		self.buffer = RolloutBuffer()

		#self.policy_old = ActorCritic(state_dim, action_dim).to(device)
		#self.policy_old.load_state_dict(self.policy.state_dict())

		self.MseLoss = torch.nn.MSELoss()

	def select_actions(self, state_batch):

		with torch.no_grad():
			actions, action_logprobs = self.policy.act(state_batch)

		self.buffer.states.append(state_batch)
		self.buffer.actions.append(actions)
		self.buffer.logprobs.append(action_logprobs)

		return actions

	def update(self):

		# Monte Carlo estimate of returns
		rewards = []
		for reward, are_done in zip(reversed(self.buffer.rewards), reversed(self.buffer.are_done)):
			if are_done:
				discounted_reward = torch.zeros(reward.shape, device=self.device)
			discounted_reward = reward + (self.discount * discounted_reward)
			rewards.insert(0, discounted_reward)

		print(discounted_reward.mean())

		# Normalizing the rewards

		rewards = torch.cat(rewards).squeeze()
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = StateTSP.from_state_buffer(self.buffer.states)
		#old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
		old_actions = torch.squeeze(torch.cat(self.buffer.actions, dim=0)).detach().to(self.device)
		old_logprobs = torch.squeeze(torch.cat(self.buffer.logprobs, dim=0)).detach().to(self.device)


		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, dist_entropy = self.policy.evaluate(old_states, old_actions)
			state_values = self.critic(old_states)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2).mean() - 0.01*dist_entropy.mean()

			# POSSIBLE PROBLEMS?!
			# I am doing Monte Carlo but the gradients only flow through single steps
			# planning the whole tour towards the last step is crucial as it might become the most expensive
			# it would be best if the gradients would be connected all the way through the whole tour creation
			# is this the case in standard attention solution?
			# in my case here logprobs definitely comes from single transitions
			# BUT monte carlo should help me realize here that easly decisions are already quite bad ... 
			# this should be learnable quite well for the value estimator
			# so my main issue is probably that the value estimator can't estimate properly
			# can i visualize value estimator observations and decisions?

			# why is prev_a not used in attention_model? i somehow need to include it when creating the context no?
			# first_a is used, current_node is used instead of prev_a, but as index, right? maybe I can do this too?

			# also I just edited prev_a and first_a to be array of -1 at the start. now i get errors
			# check _get_parallel_step_context TSP part of attention_model again


			critic_loss = self.MseLoss(state_values, rewards)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# take gradient step
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		# Copy new weights into old policy
		#self.policy_old.load_state_dict(self.policy.state_dict())

		# clear buffer
		self.buffer.clear()