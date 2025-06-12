#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  18:54
import math
import numpy as np


class TreeNode:
    def __init__(self, parent, prior, dirichlet_noise=None):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.prior = prior
        self.noise = dirichlet_noise if dirichlet_noise is not None else prior
        self.deterministic = False
    
    def train(self):
        if self.deterministic is True:
            if not self.children:
                self.deterministic = False
                return
            for node in self.children.values():
                node.train()
                self.deterministic = False

    def eval(self):
        if self.deterministic is False:
            if not self.children:
                self.deterministic = True
                return
            for node in self.children.values():
                node.eval()
                self.deterministic = True

    def expand(self, action_probs, noise=None):
        for idx, (action, prior) in enumerate(action_probs):
            if action not in self.children:
                if noise is None or self.deterministic:
                    self.children[action] = TreeNode(self, prior, prior)
                else:
                    self.children[action] = TreeNode(self, prior, noise[idx])

    def select(self, c_init, c_base):
        return max(self.children.items(), key=lambda action_node: action_node[1].PUCT(c_init, c_base))

    def update(self, leaf_value, discount):
        if self.parent:
            self.parent.update(-leaf_value * discount, discount)
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits # Q = ((n-1)*Q_old + leaf_value)/n

    def PUCT(self, c_init, c_base):
        if self.parent is not None and self.parent.is_root() and not self.deterministic:
            prior = 0.75 * self.prior + 0.25 * self.noise
        else:
            prior = self.prior
        self.u = (c_init + math.log((1 + self.parent.n_visits + c_base) / c_base)) * prior * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u
    
    def is_leaf(self):
        return not self.children
    
    def is_root(self):
        return self.parent is None


class MCTS:
    def __init__(self, policy_value_fn, c_init=1, n_playout=1000, random_flip=True):
        self.root = TreeNode(None, 1, None)
        self.policy = policy_value_fn
        self.c_init = c_init
        self.c_base = n_playout / 800 * 19652
        self.n_playout = n_playout
        self.random_flip = random_flip
    
    @property
    def Q(self):
        return self.root.Q

    def train(self):
        self.root.train()

    def eval(self):
        self.root.eval()

    def select_leaf_node(self, env):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_init, self.c_base)
            env.step(action)
        return node

    def playout(self, env, discount=1):
        node = self.select_leaf_node(env)
        env_aug, flipped = env.random_flip()
        action_probs, leaf_value = self.policy(env_aug)
        if flipped:
            action_probs = [(env.flip_action(action), prob) for action, prob in action_probs]
        if not env.done():
            node.expand(action_probs)
        node.update(-leaf_value, discount)

    def get_action(self, env, discount=1):
        for _ in range(self.n_playout):
            self.playout(env.copy(), discount)
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
        env_aug, flipped = env.random_flip()
        action_probs, leaf_value = self.policy(env_aug)
        if flipped:
            action_probs = [(env.flip_action(action), prob) for action, prob in action_probs]
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
            self.playout(env.copy(), dirichlet_alpha, discount)
        act_visits = [(action, node.n_visits)
                      for action, node in self.root.children.items()]
        actions, visits = zip(*act_visits)
        return actions, visits
    
    def greedy_backup_value(self, env, discount=1.0):
        node = self.root
        player_turn = env.turn
        while not node.is_leaf():
            best_action = max(node.children.items(), key=lambda kv: kv[1].n_visits)[0]
            env.step(best_action)
            node  = node.children[best_action]

        if env.done():
            winner = env.winPlayer()
            leaf_value = 0 if winner == 0 else (1 if winner == player_turn else -1)
        else:
            _, leaf_value = self.policy(env)
        return float(leaf_value * discount)
