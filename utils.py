#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 21/Jul/2024  03:52
import numpy as np
from operator import itemgetter


class Elo:
    def __init__(self, init_A=1500, init_B=1500):
        self.R_A = init_A
        self.R_B = init_B

    def update(self, result_a, k=32):
        """
        :param result_a: 1 (A win), 0.5 (draw), 0 (A lose).
        :param k: factor, default: 32.
        """
        expected_a = 1 / (1 + pow(10, (self.R_B - self.R_A) / 400))
        expected_b = 1 / (1 + pow(10, (self.R_A - self.R_B) / 400))
        self.R_A = max(self.R_A + k * (result_a - expected_a), 1500)
        self.R_B = max(self.R_B + k * ((1 - result_a) - expected_b), 1500)
        return self.R_A, self.R_B


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
    return list(zip(valid, action_probs)), evaluate_rollout(env.copy())


def rollout_policy_fn(env):
    valid = env.valid_move()
    probs = np.random.rand(len(valid))
    return list(zip(valid, probs))


def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)
