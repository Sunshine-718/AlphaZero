#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jun/2025  21:55
import math
import torch
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

    @property
    def is_leaf(self):
        return not self.children

    @property
    def is_root(self):
        return self.parent is None

    def PUCT(self, c_init, c_base):
        eps = 0.25
        prior = (1 - eps) * self.prior + eps * self.noise
        if self.n_visits == 0:
            self.u = float('inf')
        else:
            self.u = (c_init + math.log((1 + self.parent.n_visits + c_base) / c_base)
                      ) * prior * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return -self.Q + self.u

    def UCT(self, c_init, c_base):
        if self.n_visits == 0:
            self.u = float('inf')
        else:
            self.u = (c_init + math.log((1 + self.parent.n_visits + c_base) / c_base)
                      ) * math.sqrt(math.log(self.parent.n_visits) / self.n_visits)
        return -self.Q + self.u

    def UCB(self, c_init, c_base, UCT=False):
        if UCT:
            return self.UCT(c_init, c_base)
        else:
            return self.PUCT(c_init, c_base)

    def expand(self, action_probs, noise=None):
        for idx, (action, prior) in enumerate(action_probs):
            if action not in self.children:
                if self.deterministic or noise is None:
                    self.children[action] = TreeNode(self, prior, None)
                else:
                    self.children[action] = TreeNode(self, prior, noise[idx])

    def select(self, c_init, c_base, UCT=False):
        return max(self.children.items(), key=lambda action_node: action_node[1].UCB(c_init, c_base, UCT))

    def update(self, leaf_value):
        if self.parent:
            self.parent.update(-leaf_value)
        self.n_visits += 1
        # Q = ((n - 1) * Q_old + leaf_value) / n
        self.Q += (leaf_value - self.Q) / self.n_visits


class MCTS:
    def __init__(self, policy_value_fn, c_init=1.5, n_playout=1000, alpha=None):
        self.root = TreeNode(None, 1, None)
        self.policy = policy_value_fn
        self.c_init = c_init
        self.c_base = 500
        self.n_playout = n_playout
        self.alpha = alpha
        self.deterministic = False

    def train(self):
        self.root.train()
        self.deterministic = False

    def eval(self):
        self.root.eval()
        self.deterministic = True

    def select_leaf_node(self, env, UCT=False):
        node = self.root
        while not node.is_leaf:
            action, node = node.select(self.c_init, self.c_base, UCT)
            env.step(action)
        return node

    def playout(self, env, alpha=None):
        node = self.select_leaf_node(env, True)
        env_aug, flipped = env.random_flip()
        action_probs, leaf_value = self.policy(env_aug)
        if flipped:
            action_probs = [(env.flip_action(action), prob) for action, prob in action_probs]
        if not env.done():
            node.expand(action_probs)
        node.update(leaf_value)

    def get_action(self, env):
        for _ in range(self.n_playout):
            self.playout(env.copy())
        return max(self.root.children.items(), key=lambda action_node: action_node[1].n_visits)

    def prune_root(self, node_index):
        if node_index in self.root.children:
            self.root = self.root.children[node_index]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1)


class MCTS_AZ(MCTS):
    def playout(self, env):
        noise = None
        node = self.select_leaf_node(env)
        env_aug, flipped = env.random_flip()
        #
        valid = env_aug.valid_move()
        current_state = torch.from_numpy(env_aug.current_state()).float().to(self.policy.device)
        probs, value = self.policy.predict(current_state)
        action_probs = tuple(zip(valid, probs.flatten()[valid]))
        leaf_value = value.flatten()[0]
        #
        if flipped:
            action_probs = [(env.flip_action(action), prob) for action, prob in action_probs]
        if not env.done():
            if self.alpha is not None and not self.deterministic:
                noise = np.random.dirichlet([self.alpha for _ in action_probs])
            node.expand(action_probs, noise)
        else:
            winner = env.winPlayer()
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = (1 if winner == env.turn else -1)
        node.update(leaf_value)

    def get_action_visits(self, env):
        assert ((self.alpha is not None) or (self.alpha is None and self.deterministic))
        for _ in range(self.n_playout):
            self.playout(env.copy())
        act_visits = [(action, node.n_visits) for action, node in self.root.children.items()]
        return zip(*act_visits)
