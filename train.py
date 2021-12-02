import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to

from tianshou.data import ReplayBuffer, Batch
from tianshou.policy import SACPolicy
from problems.tsp.state_tsp import StateTSP
from torch.distributions import Categorical


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


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()


# MARK: SAC Functions ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ReplayBuffer(size=9)
# buf.add(obs=i, act=i, rew=i, done=i, obs_next=i + 1, info={})
def train_epoch_sac(
    model, 
    actor, # tianshou.policy.BasePolicy (s -> logits)
    actor_optim,
    critic1, # (s, a -> Q(s, a))
    critic1_optim,
    critic2, # (s, a -> Q(s, a))
    critic2_optim,
    value_model,
    v_optimizer,
    #sac_model: SACPolicy, 
    buffer: ReplayBuffer, 
    optimizer, baseline, 
    lr_scheduler, 
    epoch, 
    val_dataset, 
    problem, 
    tb_logger, 
    opts
):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train() # TODO check if this is necessary
    critic1.train()
    critic2.train()
    value_model.train()
    #sac_model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        collect_experience_sac(
            model, # todo use random policy or sac model
            buffer,
            problem,
            batch,
            opts
        )

        discount = 1.0
        experience_batch_size = 128
        experience_batch = sample_buffer(buffer, experience_batch_size)
        
        #print(experience_batch.obs.loc)
        #print(type(experience_batch.obs.loc))
        #print(experience_batch.obs.loc.shape)
        
        #print(state.loc.shape)

        # need a way to convert the experience_batch.obs to StateTSP...
        #print(experience_batch.obs[0])

        train_batch_sac(
            model,
            actor, # tianshou.policy.BasePolicy (s -> logits)
            actor_optim,
            critic1, # (s, a -> Q(s, a))
            critic1_optim,
            critic2, # (s, a -> Q(s, a))
            critic2_optim,
            value_model,
            v_optimizer,
            #sac_model,
            discount,
            epoch, # this should be fine
            batch_id, # this should be fine too
            step, # todo fix this value
            experience_batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

def sample_buffer(buffer: ReplayBuffer, count):
    batch, ids = buffer.sample(batch_size=count)

    #print(batch)
    #print(type(batch))
    #print(batch.obs_next.i)
    return batch


def collect_experience_sac(
        model,
        buffer: ReplayBuffer,
        problem,
        input_batch,
        opts
):
    print("collecting experience")
    
    x = input_batch # TODO remove baseline stuff from batch - not needed anymore in sac - some baselines use batch['data'] here
    x = move_to(x, opts.device)

    prev_state = problem.make_state(x)
    #prev_state.loc = move_to(prev_state.loc, opts.device)

    j = 0
    while not prev_state.all_finished():
        # Input to model will not be just a graph anymore, but rather an observation -> partially solved graph, i.e. state
        # try to input a tianshou batch here instead of a tsp state
        # first try to only use the state, without x, since x should be in state loc
        logits, _ = model(prev_state)
        # logits = torch.tensor(logits, device=opts.device) # TODO hacky for now because exp only works with tensors but tianshou wants tuples
        # logits.exp() gives probabilities
        mask = prev_state.get_mask()
        selected = model._select_node(logits.exp(), mask[:, 0, :])
        state = prev_state.update(selected)
        cost = problem.get_step_cost(prev_state, state)

        for i in range(x.shape[0]): # iterate over the whole batch and save each last step to replay buffer
            s = prev_state[torch.tensor(i)]
            s_next = state[torch.tensor(i)]

            # Note: batch expects tensors at the leaves. converting states to dicts with tensors at leaves
            # see https://tianshou.readthedocs.io/en/master/tutorials/batch.html
            obs = s.asdict()
            obs_next = s_next.asdict()

            b = Batch(obs=obs, act=s_next.prev_a, rew=-cost[i].item(), done=s_next.all_finished(), obs_next=obs_next, info={})
            # buffer doesnt like getting None once and then actual elements later
            buffer.add(b)

        prev_state = state
        print(j)
        j += 1


def train_batch_sac(
        model,
        actor, # tianshou.policy.BasePolicy (s -> logits)
        actor_optim,
        critic1, # (s, a -> Q(s, a))
        critic1_optim,
        critic2, # (s, a -> Q(s, a))
        critic2_optim,
        value_model,
        v_optimizer,
        #sac_model: SACPolicy,
        discount,
        epoch, # this should be fine
        batch_id, # this should be fine too
        step, # todo fix this value
        experience_batch: Batch,
        tb_logger,
        opts
):
    # TODO
    print("training SAC")
    #model(experience_batch)
    # sac_model.learn(experience_batch) # somehow cant deal with actual batches
    # somehow tianshou expects only single elements in forward
    # also wants logits as tuple for some reason
    # also creates a distribution using this tuple, which probably fails because logits tuple has wrong format
    # (is trying to unpack in the Normal constructor parameters)

    
    logits, _ = actor(experience_batch.obs)
    #print(logits.exp())
    m = Categorical(logits.exp())
    actions = m.sample()
    #print(actions)
    log_probs = m.log_prob(actions)
    print(log_probs.shape)

    # i dont really want to use actor for prediction here, right? this is not the idea of off-policy learning
    # so how do i get the log_probs and how do i optimize the actor?
    # for log_probs i have to use current policy acc to paper... but whyy?
    # for optimizing policy, i should probably focus on optimizing the probs somehow...



    #print(obs_result.log_prob.shape)
    current_q1 = critic1(experience_batch.obs, experience_batch.act).flatten()
    current_q2 = critic2(experience_batch.obs, experience_batch.act).flatten()
    print(current_q1.shape)
    print(current_q2.shape)
    current_v = value_model(experience_batch.obs).flatten()
    next_v = value_model(experience_batch.obs_next).flatten()
    print(next_v.shape)
    print(next_v.shape)
    current_rewards = torch.tensor(experience_batch.rew).to(opts.device).flatten() # TODO dont use tianshou batch, so i dont have to move back and fourth between cpu and gpu
    print(type(current_rewards))
    print(current_rewards.shape)
    alpha = 0.2

    # this should not effect q_networks or actor_network
    value_loss = torch.square(current_v - (torch.minimum(current_q1, current_q2) - alpha * log_probs)).mean() # logits = log_prob!?
    # these should not effect value_network
    q1_loss = torch.square(current_q1 - (current_rewards + next_v)).mean()
    q2_loss = torch.square(current_q2 - (current_rewards + next_v)).mean()

    # this should not effect q-networks
    actor_loss = - torch.minimum(current_q1, current_q2) # TODO does this work?



    # TODO prevent losses from interfering and updating wrong parameters -> use detach? i saw the trick somewhere ...











    assert(1==0)
    td = current_q - target_q
    critic_loss = (td.pow(2) * weight).mean()
    critic1_optim.zero_grad()
    critic_loss.backward()
    critic1_optim.step()
    # return td, critic_loss





    # critic 1&2
    td1, critic1_loss = self._mse_optimizer(
        batch, self.critic1, self.critic1_optim
    )
    td2, critic2_loss = self._mse_optimizer(
        batch, self.critic2, self.critic2_optim
    )
    batch.weight = (td1 + td2) / 2.0  # prio-buffer

    # actor
    obs_result = self(batch)
    a = obs_result.act
    current_q1a = self.critic1(batch.obs, a).flatten()
    current_q2a = self.critic2(batch.obs, a).flatten()
    actor_loss = (
        self._alpha * obs_result.log_prob.flatten() -
        torch.min(current_q1a, current_q2a)
    ).mean()
    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()

    if self._is_auto_alpha:
        log_prob = obs_result.log_prob.detach() + self._target_entropy
        alpha_loss = -(self._log_alpha * log_prob).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        self._alpha = self._log_alpha.detach().exp()

    self.sync_weight()

    result = {
        "loss/actor": actor_loss.item(),
        "loss/critic1": critic1_loss.item(),
        "loss/critic2": critic2_loss.item(),
    }
    if self._is_auto_alpha:
        result["loss/alpha"] = alpha_loss.item()
        result["alpha"] = self._alpha.item()  # type: ignore

    return result

    







    #result = sac_model.learn(experience_batch)
    #print(result)


    # CANT I USE TIANSHOU SAC? - only need to define the models and optimizers

    # update state-value estimator
    # gradient of: V(s) * (V(s) - Q(a|s) + log p(a|s))
    
    #state_value_loss = v_model(experience_batch.obs) * 
    #                  (v_model(experience_batch.obs) - 
    #                   q_model(experience_batch.act, experience_batch.obs) + 
    #                   log(policy_model(experience_batch.act, experience_batch.obs)))

    # update q-value estimator (TODO use two q-value estimators)
    # gradient of: Q(a|s) * (Q(a|s) - r(a|s) - gamma* V`(s+1))
    #q_value_loss = q_model(experience_batch.act, experience_batch.obs) * 
    #              (q_model(experience_batch.act, experience_batch.obs) - 
    #               experience_batch.rew - discount * v_moving(experience_batch.obs_next))

    # update policy estimator
    # KL divergence or trick ....

    # update moving average state-value


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    # TODO
    # sample batch from Replay Buffer and feed it into this function
    # adjust model(x) to work on batches of individual items from the replay buffer
    #   (only predict one step, then update models, add new experience to RB)
    #   Note: the decoder needs all of the input embeddings, this is quite inefficient, as we run per sample (maybe save it in the experience)
    # introduce Q and state value estimator and learn policy using KL divergence -> this can be done using tianshou SAC, right?
    #   the estimators will need some form of masking, might be hard to introduce
    # 

    x, bl_val = baseline.unwrap_batch(batch)
    print("---------") 
    print(x.shape) # 512, n, 2 -> 512 items, n nodes and 2 coordinates

    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)
    print("-----")
    print(cost.shape) # one batch of total costs of each trajectory
    print(log_likelihood.shape) # one batch of probabilities of each trajectory

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    print("---")
    print(bl_val) # 1 value
    print(bl_loss) # 1 value

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
