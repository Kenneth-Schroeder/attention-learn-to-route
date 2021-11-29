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

from tianshou.data import ReplayBuffer


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
def train_epoch_sac(model, buffer: ReplayBuffer, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
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

        collect_experience_sac(
            model, 
            buffer,
            batch
        )

        train_batch_sac(
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



def collect_experience_sac(
        model,
        buffer: ReplayBuffer,
        batch
):
    print("collecting experience")

    x, bl_val = baseline.unwrap_batch(batch)

    state = self.problem.make_state(x)
    print("///")
    print(state)

    # Input to model will not be just a graph anymore, but rather an observation -> partially solved graph, i.e. state

    # Compute keys, values for the glimpse MHA calculation and keys for the logits calculation once, 
    # as they can be reused in every step (only the queries change)
    fixed = self._precompute(embeddings)

    batch_size = state.ids.size(0)

    # Perform decoding steps
    i = 0
    while not (self.shrink_size is None and state.all_finished()):

        if self.shrink_size is not None:
            unfinished = torch.nonzero(state.get_finished() == 0)
            if len(unfinished) == 0:
                break
            unfinished = unfinished[:, 0]
            # Check if we can shrink by at least shrink_size and if this leaves at least 16
            # (otherwise batch norm will not work well and it is inefficient anyway)
            if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                # Filter states
                state = state[unfinished]
                fixed = fixed[unfinished]

        log_p, mask = self._get_log_p(fixed, state) # THIS IS WHERE THE MAGIC HAPPENS? it takes the precomputed fixed values and a state

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension

        state = state.update(selected)

        # Now make log_p, selected desired output size by 'unshrinking'
        if self.shrink_size is not None and state.ids.size(0) < batch_size:
            log_p_, selected_ = log_p, selected
            log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
            selected = selected_.new_zeros(batch_size)

            log_p[state.ids[:, 0]] = log_p_
            selected[state.ids[:, 0]] = selected_

        # Collect output of step
        outputs.append(log_p[:, 0, :])
        sequences.append(selected)

        i += 1

    # Collected lists, return Tensor
    return torch.stack(outputs, 1), torch.stack(sequences, 1)



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
