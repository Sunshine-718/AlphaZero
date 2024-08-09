#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  19:36
import numpy as np
from copy import deepcopy
from MCTS import MCTS

def softmax(x):
    probs = np.exp(x - np.max(x))
    return probs / np.sum(probs)

class MCTS_AZ(MCTS):
    def playout(self, env, dirichlet_alpha=0.3):
        node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            if dirichlet_alpha is not None:
                noise = np.random.dirichlet([dirichlet_alpha for _ in action_probs])
            else:
                noise = None
            node.expand(action_probs, noise)
        else:
            winner = env.winPlayer()
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = (1 if winner == env.turn else -1)
        node.update(-leaf_value)

    def get_action_visits(self, env, dirichlet_alpha=0.3):
        for _ in range(self.n_playout):
            self.playout(deepcopy(env), dirichlet_alpha)
        act_visits = [(action, node.n_visits) for action, node in self.root.children.items()]
        actions, visits = zip(*act_visits)
        return actions, visits


class AlphaZeroPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS_AZ(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def train(self):
        self.mcts.train()
    
    def eval(self):
        self.mcts.eval()
    
    def reset_player(self):
        self.mcts.update_with_move(-1)
    
    def get_action(self, env, temp=0, dirichlet_alpha=0.3):
        valid = env.valid_move()
        action_probs = np.zeros((7,), dtype=np.float32)
        if len(valid) > 0:
            actions, visits = self.mcts.get_action_visits(env, dirichlet_alpha)
            if temp == 0:
                action = max(actions, key=lambda x: visits[actions.index(x)])
            else:
                probs = softmax(np.log(np.array(visits) + 1e-8) / temp)
                action = np.random.choice(actions, p=probs)
            probs = softmax(np.log(np.array(visits) + 1e-8))
            action_probs[list(actions)] = probs
            if self.is_selfplay:
                self.mcts.update_with_move(action)
            else:
                self.mcts.update_with_move(-1)
            return action, action_probs
        else:
            print('WARNING: the board is full')
            