#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  18:54
import numpy as np
from copy import deepcopy
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


class TreeNode:
    def __init__(self, parent, prior, dirichlet_noise=None, deterministic=False):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.prior = prior
        self.noise = dirichlet_noise if dirichlet_noise is not None else prior
        self.deterministic = deterministic
    
    def expand(self, action_probs, noise=None):
        noise = [self.prior for _ in action_probs] if (noise is None or self.deterministic) else noise
        for idx, (action, prior) in enumerate(action_probs):
            if action not in self.children:
                self.children[action] = TreeNode(self, prior, noise[idx], self.deterministic)
    
    def select(self, c_puct):
        return max(self.children.items(), key=lambda action_node: action_node[1].get_value(c_puct))
    
    def update(self, leaf_value):
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits
    
    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    
    def get_value(self, c_puct):
        if self.parent is not None and self.parent.is_root() and not self.deterministic:
            prior = 0.75 * self.prior + 0.25 * self.noise
        else:
            prior = self.prior
        self.u = c_puct * prior * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u
    
    def is_leaf(self):
        return self.children == {}
    
    def is_root(self):
        return self.parent is None
    
class MCTS:
    def __init__(self, policy_value_fn, c_puct=1, n_playout=10000):
        self.determinstic = False
        self.root = TreeNode(None, 1, None, self.determinstic)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout
    
    def train(self):
        self.determinstic = False
    
    def eval(self):
        self.determinstic = True
    
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
        node.update_recursive(-leaf_value)
    
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

class MCTSPlayer:
    def __init__(self, c_puct=1, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
    
    def set_player_ind(self, p):
        self.player = p
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def get_action(self, env):
        valid = env.valid_move()
        if len(valid) > 0:
            action = self.mcts.get_action(env)
            self.mcts.update_with_move(-1)
            return action
        else:
            print('Warning: the board is full')
