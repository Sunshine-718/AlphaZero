#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  18:54
import numpy as np
from copy import deepcopy
from TreeRep import TreeNode
from operator import itemgetter


def rollout_policy_fn(env):
    valid = env.valid_move()
    probs = np.random.rand(len(valid))
    return list(zip(valid, probs))


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


class MCTS:
    def __init__(self, policy_value_fn, c_puct=1, n_playout=10000):
        self.root = TreeNode(None, 1, None)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def train(self):
        self.root.train()

    def eval(self):
        self.root.eval()

    def select_leaf_node(self, env):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            env.step(action)
        return node

    def playout(self, env):
        node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            node.expand(action_probs)
        node.update(-leaf_value)

    def get_action(self, env):
        for _ in range(self.n_playout):
            self.playout(deepcopy(env))
        return max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits)

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1)
