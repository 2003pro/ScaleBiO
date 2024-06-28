#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
All rights reserved.
'''
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=no-member

from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import logging.config
import textwrap
import wandb

import numpy as np
from accelerate.logging import get_logger

class Error(Exception):
    """Root for all errors."""

class ParseError(Error):
    """Unable to parse input data."""

class UnsupportedAlgorithmError(Error):
    "The algorithm is not supported."""

class UnsupportedSchedulerError(Error):
    """The learning rate schedule is not supported."""

class UnsupportedFrameworkError(Error):
    """The framework is not supported."""

class UnsupportedModelError(Error):
    """The model type is not supported."""

class UnsupportedDatasetError(Error):
    """The dataset is not supported."""

class UnsupportedOptimizerError(Error):
    """The optimizer is not supported."""

class UnsupportedLRSchedulerError(Error):
    """The lr scheduler is not supported."""

class InvalidArgumentValueError(Error):
    """The provided argument has invalid values."""


def parse_argument(sys_argv):
    """Parses arguments from command line.

    Args:
        sys_argv: the list of arguments (strings) from command line.

    Returns:
        A struct whose member corresponds to the required (optional) variable.
        For example,
        ```
        args = parse_argument(['main.py' '--input', 'a.txt', '--num', '10'])
        args.input       # 'a.txt'
        args.num         # 10
        ```
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='A simple framework for regression/classification')

    # Framework choice
    parser.add_argument(
        '--framework', type=str, default='tensorflow',
        help=textwrap.dedent(
            '''
            Supported framework:
                1) "tensorflow": Use tensorflow to optimize ridge regression.
                2) "numpy": Use numpy + manually computed gradient to optimize.
            '''))

    # Training parameters
    parser.add_argument(
        '--train_data', type=str, required=True,
        help=textwrap.dedent(
            '''
            The training data. Each line has a format as follows:
                `{label} 1:{feature_1} 2:{feature_2} ... d:{feature_d}`
            where d is the number of features. For example,
                `15.0 1:-1 2:0.027027 3:0.0420168 4:-0.831858 5:-0.63733`
            This format follows the convention in libsvm dataset. Please refer
            to
                https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html
            for more details.
            '''))

    parser.add_argument(
        '--alpha', type=float, required=True,
        help='The regularization coefficient')
    parser.add_argument(
        '--num_epoch', type=int, required=True,
        help='The number of epochs')
    parser.add_argument(
        '--init_lr', type=float, required=True,
        help='Initial learning rate')
    parser.add_argument(
        '--lr_schedule_conf_file', type=str, required=True,
        help=textwrap.dedent(
            '''
            The config file name for different learning rate schedules.
            Its general format is as follows,
            ```
            [general]
            type = {type}

            [hyperparams]
            ...
            ```

            Supported types of learning rate schedules are as follows,
                * "inverse_time_decay":
                    During iteration t, learning rate is
                        `init_lr / (1 + init_lr * t * lambda)`
                    here lambda is a hyperparameter for this schedule.
                    Iteration t starts from 0.
                    Config example:
                        ```
                        [general]
                        type = inverse_time_decay

                        [hyperparams]
                        lambda = 0.1
                        ```

                * "piecewise_constant":
                    Specifies the starting point of each interval s_i (i>0),
                    learning rate will be,
                        `init_lr * c_i`
                    if t in [s_i, s_{i+1}). Here c_i is the factor of this
                    interval. s_0=0 and s_{n+1}=+oo by default.
                    Config example:
                        ```
                        [general]
                        type = piecewise_constant

                        [hyperparams]
                        starting_points = 100, 500, 1000
                        factors = 0.1, 0.01, 0.001
                        ```

                * "cosine_decay":
                    [2016] SGDR: STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
                    (https://arxiv.org/pdf/1608.03983.pdf)

                    Learning rate decays in each segment [0, t_max]. t_max =
                    t_0, t_cur=0 initially. After the end of each segment,
                    expand the length of that segment by multiplying `t_mul`,
                    i.e. t_max *= t_mul, then increase `t_cur` by 1, i.e. t_cur
                    += 1.
                        ```python
                        cos_decay = 0.5 * (1 + cos(pi * t_cur / t_max))
                        lr = min_lr + (init_lr - min_lr) * cos_decay
                        ```

                    Config example:
                        ```
                        [general]
                        type = cosine_decay

                        [hyperparams]
                        t_0 = 100
                        t_mul = 1.0
                        min_lr = 0.00001

                * "exponential_decay":
                    During interation t, which starts from 0 initially,
                        `lr = init_lr * (decay_rate ** (t / decay_step))`
                    Note that the division here `t / decay_step` is integer
                    division.

                    Config example:
                        ```
                        [general]
                        type = exponential_decay

                        [hyperparams]
                        decay_step = 1
                        decay_rate = 0.9999
                        ```

                * "piecewise_inverse_time":
                    Specifies the starting point of each interval s_i (i>=0),
                    learning rate will be,
                        `init_lr / (a_i * (t - s_i) + b_i)`
                    if t in [s_i, s_{i+1}). Here a_i and b_i is the
                    hyperparameter for interval. It is required that s_1=0.
                    s_{n+1}=+oo.

                    Config example:
                        ```
                        [general]
                        type = piecewise_constant

                        [hyperparams]
                        starting_points = 0, 100, 500, 1000
                        a = 1.0, 2.0, 4.0, 8.0
                        b = 0.0, 0.0, 0.0, 0.0
                        ```
            '''))

    # Debug parameters
    parser.add_argument(
        '--pseudo_random', const=True, default=False, nargs='?',
        help='A global option to make all random operations deterministic')
    parser.add_argument(
        '--logging_conf_file', default='conf/common.log_conf', type=str,
        help='The configuration file for logging')

    # Parses from commandline
    args = parser.parse_args(sys_argv[1:])

    return args


def parse_lr_schedule_conf(conf):
    """Parses the config for learning rate schedule.

    Args:
        conf: a ConfigParser object, which stores raw config information of the
            learning rate schedule.
    Returns:
        a function which computes the current learning rate,
            ```
            learning_rate = lr_scheduler(init_lr, t)
            ```
        function arguments:
            init_lr: float, the intial learning rate;
            t: int, current iteration number, starts from 0, i.e. we have t=0
                for first SGD update.
    Raises:
        ParseError if hyperparameters are invalid.
        UnsupportedSchedulerError if the scheduler type is not found.
    """
    schedule_type = conf.get('general', 'type')
    logging.info('lr schedule: type = %s', schedule_type)

    if schedule_type == 'inverse_time_decay':
        return get_inverse_time_decay_scheduler(conf)
    if schedule_type == 'piecewise_constant':
        return get_piecewise_constant_scheduler(conf)
    if schedule_type == 'cosine_decay':
        return get_cosine_decay_scheduler(conf)
    if schedule_type == 'exponential_decay':
        return get_exponential_decay_scheduler(conf)
    if schedule_type == 'piecewise_inverse_time':
        return get_piecewise_inverse_time_scheduler(conf)
    if schedule_type == 'continuous_eigencurve':
        return get_continuous_eigencurve_scheduler(conf)
    if schedule_type == 'poly_remain_time_decay':
        return get_poly_remain_time_decay_scheduler(conf)
    if schedule_type == 'elastic_step_decay':
        return get_elastic_step_decay_scheduler(conf)
    if schedule_type == 'step_decay':
        return get_step_decay_scheduler(conf)

    raise UnsupportedSchedulerError(
        'unsupported learning rate schedule "%s"' % schedule_type)


def parse_sample(line, feature_dim=0):
    """Parses a single line, which represents a sample.

    Format: `{label} 1:{feature_1} 2:{feature_2} ... d:{feature_d}`
    Example: `15.0 1:-1 2:0.027027 3:0.0420168 4:-0.831858 5:-0.63733`

    Args:
        line: str, raw data of a sample.
        feature_dim: int, feature dimension size. 0 means using number of
            available features as the feature dimension size.
    Returns:
        A tuple (feature, label), where 'feature' is a numpy array of floats,
        label is a float.
    Raises:
        ParseError if any error occurs during parsing.
    """
    # '15.0 1:-1 2:0.027027 3:0.0420168 4:-0.831858 5:-0.63733'
    # -> ['15.0', '1:-1 2:0.027027', ..., '5:-0.63733']
    array = line.split()
    label = float(array[0])

    # First scan: gets the minimum valid feature dimension
    index_set = set()
    for i, raw_single_feature in enumerate(array[1:]):
        # '2:0.027027' -> ['2', '0.027027']
        splited_raw = raw_single_feature.split(':')

        if len(splited_raw) != 2:
            raise ParseError(
                '%d-th entry in the feature does not conform the format' % i)

        index = int(splited_raw[0])
        if index in index_set:
            raise ParseError(
                '%d-th entry in the feature has a duplicated index' % i)
        index_set.add(index)

        index -= 1              # In libsvm, index starts from 1
        feature_dim = max(feature_dim, index + 1)

    feature_dim = max(len(array) - 1, feature_dim)
    feature = np.zeros(feature_dim)     # Default value = 0 for missing features

    # Second scan: gets the feature content
    for i, raw_single_feature in enumerate(array[1:]):
        # '2:0.027027' -> ['2', '0.027027']
        splited_raw = raw_single_feature.split(':')

        index = int(splited_raw[0]) - 1
        value = float(splited_raw[1])
        feature[index] = value

    return feature, label


def load_input(file_path):
    """Loads input data from path 'input_file'.

    Input data should have the following format each line:
        `{label} 1:{feature_1} 2:{feature_2} ... d:{feature_d}`
    where d is the number of features. For example,
        `15.0 1:-1 2:0.027027 3:0.0420168 4:-0.831858 5:-0.63733`

    Args:
        file_path: str, which specifies the path of the input file.

    Returns:
        A tuple of numpy arrays (feature_matrix, label_array), where
        'feature_matrix' is a Nxd 2D numpy array of floats, 'label_array' is a
        1-D numpy array of floats with size N. Here N stands for number of
        samples, d stands for feature dimension size.

    Raises:
        ParseError if any error occurs during parsing.
    """
    n = 0
    d = 0
    # Gets sample number and feature dimension size
    with open(file_path, 'r') as input_file:
        for line in input_file:
            n += 1
            feature, _ = parse_sample(line)
            d = max(d, len(feature))

    # Parses data
    feature_matrix = np.zeros(shape=(n, d))
    label_array = np.zeros(n)
    with open(file_path, 'r') as input_file:
        for i, line in enumerate(input_file):
            try:
                feature_matrix[i], label_array[i] = parse_sample(line, d)
            except ParseError as ex:
                raise ParseError('Feature %d: %s' % (i, ex.message))

    return feature_matrix, label_array


def print_runtime_statistics(t, stat, w, loss):
    """Prints statistics we concerned during run time.

    Args:
        t: int, current iteration number, starting from 0, e.g. 0 means the
            first iteration just completed.
        stat: a dict, which maps names to relevant precomputed ridge regression
            statistics.
        w: a 1-D numpy of floats, which stands for current weights for ridge
            regression model.
        loss: a float specifies the current loss.
    """
    n = stat['num_sample']
    d = stat['feature_dimension']
    p = stat['rotation_matrix']
    rotated_w = np.matmul(p, w)
    rotated_optimum = stat['rotated_optimum']
    rotated_dist = np.abs(rotated_w - rotated_optimum)
    logging.debug('----- iter %d: distance to w* in rotated space: %s',
                  t, str(rotated_dist.tolist()))

    # Portions each part contributes to the loss
    eigen_values = stat['eigen_values']
    loss_each_dim = eigen_values * (rotated_dist ** 2)
    sum_loss_dist = loss_each_dim.sum()
    avg_loss_dist = sum_loss_dist / n
    logging.debug('----- iter %d: average loss distance to loss*: %.10f',
                  t, loss - stat['optimal_loss'])
        # Notice that `avg_loss - loss* == avg_loss_dist`, where,
        #    avg_loss is the average loss over the whole dataset;
        #    loss* is the optimal loss over the whole dataset;
        #    avg_loss_dist is the sum of losses for each dimension in
        #           the rotated space divided by n.

    if sum_loss_dist > 1e-6:
        logging.debug(
            '----- iter %d: loss portion each dimension in rotated space: %s',
            t, str((loss_each_dim / sum_loss_dist).tolist()))
    else:
        logging.debug(
            '----- iter %d loss portion each dimension in rotated space: %s',
            t, str((np.full(d, 1 / d)).tolist()))


def logging_stat_dict(stat_dict, prefix='', suffix='', use_wandb=False, accelerator=None):
    logger = get_logger('accelerator')
    stat_str_list = [f'{prefix}']
    for key, value in stat_dict.items():
        stat_str_list.append(f' {key} = {value},')
    stat_str_list.append(f'{suffix}')

    stat_str = ''.join(stat_str_list)
    logger.info(stat_str)

    if use_wandb and accelerator.is_main_process:
        wandb.log(stat_dict)
