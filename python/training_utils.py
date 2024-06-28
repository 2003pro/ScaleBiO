#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel

from utils import UnsupportedOptimizerError

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(last=None)
def get_val_sampler(
    data_loader,
    data_loader_iter=None,
    sampler_type='stochastic',
    reuse_last_samples=False
):
    """Chooses sampler."""
    if sampler_type == 'full':
        return data_loader
    elif sampler_type == 'stochastic':
        if not reuse_last_samples:
            if data_loader_iter is not None:
                get_val_sampler.last = [next(data_loader_iter)]
            else:
                raise ValueError('data loader iter cannot be None')
        return get_val_sampler.last

@static_vars(last=None, cache=[])
def get_train_sampler(
    data_loader,
    data_loader_iter=None,
    sampler_type='stochastic',
    reuse_last_samples=False
):
    """Chooses sampler."""
    if sampler_type == 'full':
        return data_loader
    elif sampler_type == 'stochastic':
        if not reuse_last_samples:
            if data_loader_iter is not None:
                get_train_sampler.last = [next(data_loader_iter)]
                get_train_sampler.cache.append(get_train_sampler.last)
            else:
                raise ValueError('data loader iter cannot be None')
            return get_train_sampler.last
        else:
            if len(get_train_sampler.cache) == 0:
                raise ValueError('Reuse last sample too many times!')
            smp=get_train_sampler.cache.pop(0)
            return smp


def loss_on_batch(
    model,
    samp_prob_p,
    accelerator,
    x_batch,
    y_batch,
    p_batch,
    attn_mask,
    retain_grad=False,
    loss_scale_factor: float = 1.0,
):
    """Crossentropy_loss(model(x_batch), y_batch) + regularizer(model)."""
    
    per_partition_loss=torch.zeros(3)
    if samp_prob_p is not None:
        predicted_logits = model(input_ids=x_batch, attention_mask=attn_mask, return_dict=True).logits#.to(torch.float16)

        def loss_per_token(logits, labels):
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            return loss
    
        crossent_loss=loss_per_token(predicted_logits, y_batch)
        crossent_loss=crossent_loss.reshape(y_batch.shape[0], -1).mean(dim=-1)#.to(torch.float16)
        loss=samp_prob_p(crossent_loss, p_batch)

        w_nj=1.0/torch.bincount(p_batch)[p_batch].cuda()#.to(torch.float16)
        crossent_loss=torch.mul(w_nj, crossent_loss)#.to(torch.float16)

        # with FullyShardedDataParallel.summon_full_params(samp_prob_p, with_grads=True):
        per_partition_loss=torch.zeros_like(samp_prob_p.prob).cuda()#.to(torch.float32)
        per_partition_loss.index_add_(0, p_batch.cuda(), crossent_loss)
    else:
        loss = model(x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True).loss

    loss = loss_scale_factor * loss
    accelerator.backward(loss)

    per_partition_loss = loss_scale_factor * per_partition_loss

    if not retain_grad:
        model.zero_grad()
    # mean_loss=accelerator.gather(loss).mean().item()
    return loss, per_partition_loss


def loss_on_sampler(
    model,
    samp_prob_p,
    accelerator,
    data_loader,
    data_loader_iter,
    sampler_type,
    reuse_last_samples,
    id_str,
    args,
    loss_scale_factor: float = 1.0,
    return_grad: bool = True,
    zero_grad: bool = True,
    mode: str = 'train',
):
    """Gets loss over a sampler.

    Arguments:
        reuse_last_samples: bool, whether we reuse the sampler, and make it the
            same as last call of `loss_on_sampler`.
    """
    if mode == 'train':
        model.train()    # Takes care of special model behaivors like dropout, BN
    elif mode == 'eval':
        model.eval()
    else:
        raise ValueError(f'Unsupported model mode "{mode}"')

    if zero_grad:
        model.zero_grad()

    sum_loss = 0.0
    if mode == 'train' and id_str!='L1(w, lambd)':
        sampler = get_train_sampler(
            data_loader,
            data_loader_iter,
            sampler_type,
            reuse_last_samples,
        )
    elif mode == 'eval' or id_str=='L1(w, lambd)':
        sampler = get_val_sampler(
            data_loader,
            data_loader_iter,
            sampler_type,
            reuse_last_samples,
        )
    else:
        raise ValueError(f'Unsupported model mode "{mode}"')
    p_batch=None
    per_partition_loss=None
    for batch in sampler:
        x_batch=batch['input_ids']
        y_batch=batch['labels']
        p_batch=batch['source']
        attn_mask=batch['attention_mask']
        x_batch = x_batch.to(accelerator.device)
        y_batch = y_batch.to(accelerator.device)
        attn_mask = attn_mask.to(accelerator.device)

        batch_loss, per_partition_loss = loss_on_batch(
            model,
            samp_prob_p,
            accelerator,
            x_batch,
            y_batch,
            p_batch,
            attn_mask,
            retain_grad=True,
            loss_scale_factor=loss_scale_factor,
        )
        sum_loss += batch_loss
   
    num_batch = len(sampler)
    assert num_batch == 1
    mean_loss = sum_loss / float(num_batch)

    # Acquires accumulated gradient, notice that the last batch may have
    # incorrect gradient (more sample weights), but commonly this is not a big
    # issue.
    return mean_loss, per_partition_loss

@torch.no_grad()
def evaluate(model, accelerator, data_loader, args):
    """Evaluates model performance on dataset `data_loader`."""
    torch_rng_state = torch.get_rng_state()

    model.eval()
    sum_loss = 0.

    for batch in data_loader:
        batch=batch.to(accelerator.device)
        x_batch=batch['input_ids']
        y_batch=batch['labels']
        attn_mask=batch['attention_mask']
        x_batch = x_batch# .cuda()
        y_batch = y_batch# .cuda()
        attn_mask = attn_mask# .cuda()

        loss = model(input_ids=x_batch, labels=y_batch, attention_mask=attn_mask, return_dict=True).loss
        sum_loss += accelerator.gather(loss).detach().cpu().mean()

    num_batch = len(data_loader)
    eval_loss = sum_loss / float(num_batch)
    eval_ppl=torch.exp(eval_loss)

    # Resets the random state, we don't want evaluation to affect the
    # reproducibility of the main training process! However, even we use
    # separated dataloaders (even unshuffled), they can affect the random state
    # (https://github.com/pytorch/pytorch/issues/11062). So we have to reset
    # torch's random state.
    torch.set_rng_state(torch_rng_state)
    return eval_loss, eval_ppl


def update_optimizer_lr(optimizer, learning_rate):
    """Updates the global learning rate of an optimizer."""
    for g in optimizer.param_groups:
        g['lr'] = learning_rate


def get_optimizer(model_params, optimizer_args_dict):
    """Gets optimizer given a configuration.

    Args:
        model_params: same type as model.parameters().
        optimizer_args_dict: a dict mapping optimizer arguments to their values.
            Except one key called "name", which specifies the optimizer name.
    """
    name = optimizer_args_dict['name']
    new_optimizer_args_dict = copy.deepcopy(optimizer_args_dict)
    new_optimizer_args_dict.pop('name', None)

    if name == 'sgd':
        return torch.optim.SGD(model_params, **new_optimizer_args_dict)
    elif name == 'adagrad':
        return torch.optim.Adagrad(model_params, **new_optimizer_args_dict)
    elif name == 'adam':
        return torch.optim.Adam(model_params, **new_optimizer_args_dict)
    elif name == 'adamw':
        return torch.optim.AdamW(model_params, **new_optimizer_args_dict)
    else:
        raise UnsupportedOptimizerError(f'Optimizer "{name}" is not supported')


def get_cosine_decay_scheduler(lr_scheduler_args_dict):
    if not 'min_lr' in lr_scheduler_args_dict:
        raise ValueError(
            f'min_lr must be provided, e.g. "min_lr=0.0"'
        )
    min_lr = lr_scheduler_args_dict['min_lr']

    def cosine_decay_scheduler(init_lr, current_step, num_steps):
        if init_lr < min_lr:
            raise ValueError(
                f'initial lr "{init_lr}"'
                f' must be no less than "{min_lr}"'
            )
        lr_factor = 0.5 * (1 + np.cos(np.pi * current_step / num_steps))
        lr = (init_lr - min_lr) * lr_factor + min_lr
        return lr

    return cosine_decay_scheduler


def get_exponential_decay_scheduler(lr_scheduler_args_dict):
    if not 'min_lr' in lr_scheduler_args_dict:
        raise ValueError(
            f'min_lr must be provided, e.g. "min_lr=0.0"'
        )
    min_lr = lr_scheduler_args_dict['min_lr']

    def exponential_decay_scheduler(init_lr, current_step, num_steps):
        if init_lr < min_lr:
            raise ValueError(
                f'initial lr "{init_lr}"'
                f' must be no less than "{min_lr}"'
            )

        lr_factor = (min_lr / init_lr) ** (current_step / num_steps)
        lr = init_lr * lr_factor
        return lr

    return exponential_decay_scheduler


def get_lr_scheduler(lr_scheduler_args_dict):
    """Gets learning rate scheduler given a configuration.

    Args:
        lr_scheduler_args_dict: a dict mapping lr scheduler arguments to
            their values. Except one key called "name", which specifies the
            lr scheduler name.
    """
    name = lr_scheduler_args_dict['name']

    if name == 'cosine' or name == 'cos':
        return get_cosine_decay_scheduler(lr_scheduler_args_dict)
    elif name == 'exponential' or name == 'exp':
        return get_exponential_decay_scheduler(lr_scheduler_args_dict)
    else:
        raise UnsupportedLRSchedulerError(
            f'LR scheduler "{name}" is not supported'
        )