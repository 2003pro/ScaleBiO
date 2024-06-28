#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=self-assigning-variable
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=literal-comparison

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import logging.config
import os
import sys
import wandb
from functools import partial
from configparser import ConfigParser
from parse_args import parse_argument, parse_args_dict

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from minmax import MinmaxConfig
from minmax import Minmax


import transformers
from accelerate import Accelerator
from data import JsonDataset, tokenize_prompt, tokenize_conversion, tokenize_text_only
from prompt_maker import PromptMaker


def get_dataset(json_data_dir: str, response_loss_only, tokenizer, max_length, sharegpt_format: bool, pretrain: bool):
    if pretrain:
        transform=partial(tokenize_text_only, response_loss_only=response_loss_only, max_length=max_length, tokenizer=tokenizer)
    elif sharegpt_format:
        transform=partial(tokenize_conversion, response_loss_only=response_loss_only, max_length=max_length, tokenizer=tokenizer)
    else:
        transform=partial(tokenize_prompt, response_loss_only=response_loss_only, max_length=max_length, tokenizer=tokenizer, prompt_maker=PromptMaker())

    dataset=JsonDataset(json_data=json_data_dir, 
                                 shuffle=True, train=True,
                                 transform=transform,
                                 chunk_long_text=pretrain)
    return dataset

def load_data(
    args,
    train_sample_percentage: float=0,
    validation_sample_percentage: float=0,
):
    """Loads data in pytorch.

    Args:
        dataset_name: str. Supported datasets ['cifar10', 'cifar100'].
        batch_size: int.
        train_sample_percentage: float.
        validation_sample_percentage: float.
        args: The parsed commandline arguments.
    Returns:
        (train_loader, test_loader, stat), where,
            * train_loader: a DataLoader object for loading training data.
            * test_loader, a DataLoader object for loading test data.
            * stat_dict: a dict maps data statistics names to their values, e.g.
                'num_sample', 'num_class'.
    """
    # Chooses dataset
    tokenizer=transformers.AutoTokenizer.from_pretrained(args.tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = get_dataset(
        args.train_data,
        args.response_loss_only,
        tokenizer,
        max_length=args.max_length,
        sharegpt_format=args.sharegpt_format,
        pretrain=args.pretrain
    )
    print(train_dataset)
    print(train_dataset[0])
    val_dataset = get_dataset(
        args.val_data,
        args.response_loss_only,
        tokenizer,
        max_length=args.max_length,
        sharegpt_format=args.sharegpt_format,
        pretrain=False
    )
    test_dataset = get_dataset(
        args.test_data,
        args.response_loss_only,
        tokenizer,
        max_length=args.max_length,
        sharegpt_format=args.sharegpt_format,
        pretrain=False
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"),
        num_workers=args.num_dataload_worker,
        batch_size=args.micro_batch_size)


    val_loader = DataLoader(
        val_dataset,
        shuffle=True,
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"),
        num_workers=args.num_dataload_worker,
        batch_size=args.micro_batch_size)
    
    val_loader_for_eval = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"),
        num_workers=args.num_dataload_worker,
        batch_size=args.val_batch_size)

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"),
        num_workers=args.num_dataload_worker,
        batch_size=args.val_batch_size)

    # Prepares necessary training information
    stat_dict = {}

    return train_loader, val_loader, val_loader_for_eval, test_loader, stat_dict


def optimize(
    train_loader,
    val_loader,
    val_loader_for_eval,
    test_loader,
    args,
    optimizer_args_dict,
    lr_scheduler_args_dict,
    accelerator
):
    """Runs optimization in pytorch.

    Args:
        train_loader: DataLoader for loading batches of training samples.
        val_loader: DataLoader for loading batches of validation samples.
        val_loader_for_eval: DataLoader for loading batches of validation
            samples. This loader exists, so that the evaluation process does
            not affect the reproducibility of the training process.
        test_loader: DataLoader for loading batches of test samples.
        image_shape: torch.Size, the shape of the image.
        num_class: int, the number of classes.

        algorithm_type: a str specifies the algorithm type.
        args: a 'class-like' object, the content parsed from commandline. It
            elements can be directly accessed in a way like 'args.num_epoch'.
        optimizer_args_dict: a dict mapping optimizer arguments to their
            values. Except one extra key "name", which maps to optimizer name.
        lr_scheduler_args_dict: a dict mapping lr scheduler arguments to
            their values. Except one extra key "name", which maps to
            lr scheduler name.

    Raises:
        UnsupportedModelError if 'model_type' not supported.
    """

    config = MinmaxConfig(
        args=args,
        optimizer_args_dict=optimizer_args_dict,
        lr_scheduler_args_dict=lr_scheduler_args_dict,
    )
    algorithm = Minmax(config)

    algorithm.run(
        train_loader=train_loader,
        val_loader_for_train=val_loader,
        val_loader_for_eval=val_loader_for_eval,
        test_loader=test_loader,
        use_wandb=args.use_wandb,
        accelerator=accelerator
    )


def main():
    """Uses deep learning models to analyze SGD with learning rate schedule."""
    # Parses arguments and loads configurations
    args = parse_argument(sys.argv)
    optimizer_args_dict = parse_args_dict(args.optimizer)
    lr_scheduler_args_dict = parse_args_dict(args.lr_scheduler)
    logging.config.fileConfig(args.logging_conf_file)
    logging.info('#################################################')
    logging.info('args = %s', str(args))

    # Controls pseudorandom behavior
    if args.pseudo_random:
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        random.seed(args.seed + 1)
        np.random.seed(args.seed + 1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f'set seed to {args.seed}')

    # Loads training/validation/test data
    train_loader, val_loader, val_loader_for_eval, test_loader, stat_dict = load_data(args=args)
    
    grad_accumulation_steps = args.global_batch_size // args.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    grad_accumulation_steps = grad_accumulation_steps // world_size
    # accelerator=Accelerator(mixed_precision='fp16',
    #                             gradient_accumulation_steps=grad_accumulation_steps) if args.fp16 else Accelerator(gradient_accumulation_steps=grad_accumulation_steps)
    accelerator=Accelerator(mixed_precision='bf16',
                                gradient_accumulation_steps=1) if args.bf16 else Accelerator(gradient_accumulation_steps=1)

    # Runs optimization
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=args
        )
    try:
        optimize(
            train_loader=train_loader,
            val_loader=val_loader,
            val_loader_for_eval=val_loader_for_eval,
            test_loader=test_loader,
            args=args,
            optimizer_args_dict=optimizer_args_dict,
            lr_scheduler_args_dict=lr_scheduler_args_dict,
            accelerator=accelerator
        )
    except Exception as e:
        if args.use_wandb and accelerator.is_main_process:
            wandb.finish()
        raise e

    if args.use_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == '__main__':
    main()