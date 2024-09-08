#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 21/Jul/2024  03:52
import numpy as np
from copy import deepcopy
from operator import itemgetter


def evaluate_rollout(env, limit=1000):
    player = env.turn
    for _ in range(limit):
        if env.done():
            break
        action_probs = rollout_policy_fn(env)
        max_action = max(action_probs, key=itemgetter(1))[0]
        env.step(max_action)
    else:
        print('Warning: rollout reached move limit.')
    winner = env.winPlayer()
    if winner == 0:
        return 0
    else:
        return 1 if winner == player else -1


def policy_value_fn(env):
    valid = env.valid_move()
    action_probs = np.ones(len(valid)) / len(valid)
    return list(zip(valid, action_probs)), evaluate_rollout(deepcopy(env))


def rollout_policy_fn(env):
    valid = env.valid_move()
    probs = np.random.rand(len(valid))
    return list(zip(valid, probs))


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)
