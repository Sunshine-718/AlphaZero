#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  18:54
import numpy as np
from copy import deepcopy
from TreeRep import TreeNode


class MCTS:
    def __init__(self, policy_value_fn, c_puct=1, n_playout=1000):
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

    def playout(self, env, discount=1):
        node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            node.expand(action_probs)
        node.update(-leaf_value, discount)

    def get_action(self, env, discount=1):
        for _ in range(self.n_playout):
            self.playout(deepcopy(env), discount)
        return max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits)

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1)


class MCTS_AZ(MCTS):
    def playout(self, env, dirichlet_alpha=0.3, discount=1):
        node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            if dirichlet_alpha is not None:
                noise = np.random.dirichlet(
                    [dirichlet_alpha for _ in action_probs])
            else:
                noise = None
            node.expand(action_probs, noise)
        else:
            winner = env.winPlayer()
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = (1 if winner == env.turn else -1)
        node.update(-leaf_value, discount)

    def get_action_visits(self, env, dirichlet_alpha=0.3, discount=1):
        for _ in range(self.n_playout):
            self.playout(deepcopy(env), dirichlet_alpha, discount)
        act_visits = [(action, node.n_visits)
                      for action, node in self.root.children.items()]
        actions, visits = zip(*act_visits)
        return actions, visits


class MCTS_AZ_SP(MCTS_AZ):
    def playout(self, env, dirichlet_alpha=0.3, discount=1):
        node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            if dirichlet_alpha is not None:
                noise = np.random.dirichlet(
                    [dirichlet_alpha for _ in action_probs])
            else:
                noise = None
            node.expand(action_probs, noise)
        else:
            leaf_value = 0
        node.update_(leaf_value, discount)
