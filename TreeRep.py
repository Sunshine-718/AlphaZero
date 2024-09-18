#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:49
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

    def select(self, c_puct):
        return max(self.children.items(), key=lambda action_node: action_node[1].PUCT(c_puct))

    def update(self, leaf_value, discount):
        if self.parent:
            self.parent.update(-leaf_value * discount, discount)
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits # Q = ((n-1)*Q_old + leaf_value)/n
    
    def update_(self, leaf_value, discount):
        if self.parent:
            self.parent.update(leaf_value * discount, discount)
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits # Q = ((n-1)*Q_old + leaf_value)/n

    def PUCT(self, c_puct):
        if self.parent is not None and self.parent.is_root() and not self.deterministic:
            prior = 0.75 * self.prior + 0.25 * self.noise
        else:
            prior = self.prior
        self.u = c_puct * prior * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u
    
    def is_leaf(self):
        return not self.children
    
    def is_root(self):
        return self.parent is None
