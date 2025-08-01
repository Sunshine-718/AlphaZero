#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47
import torch
import numpy as np
from abc import abstractmethod, ABC
from utils import policy_value_fn, softmax
from MCTS import MCTS, MCTS_AZ


class Player(ABC):
    def __init__(self):
        self.win_rate = float('nan')
        self.mcts = None

    def reset_player(self):
        pass

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def get_action(self, *args, **kwargs):
        raise NotImplementedError


class NetworkPlayer(Player):
    def __init__(self, net, deterministic=True):
        super().__init__()
        self.net = net
        self.pv_fn = self.net
        self.deterministic = deterministic

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def get_action(self, env, *args, **kwargs):
        valid = env.valid_move()
        probs = self.net.policy(env.current_state())
        action_probs = tuple(zip(valid, probs.flatten()[valid]))
        actions, probs = list(zip(*action_probs))
        probs = np.array(probs, dtype=np.float32)
        probs /= probs.sum()
        if self.deterministic:
            action = actions[np.argmax(probs)]
        else:
            action = np.random.choice(actions, p=probs)

        full_probs = np.zeros(self.net.n_actions, dtype=np.float32)
        for a, p in action_probs:
            full_probs[a] = p
        return action, full_probs


class Human(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, env, *args, **kwargs):
        move = int(input('Your move: '))
        return move, None


class MCTSPlayer(Player):
    def __init__(self, c_puct=4, n_playout=2000):
        super().__init__()
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.prune_root(-1)

    def get_action(self, env):
        action = self.mcts.get_action(env)
        self.mcts.prune_root(action)
        return action


class AlphaZeroPlayer(MCTSPlayer):
    def __init__(self, policy_value_fn, c_puct=1.5, n_playout=1000, alpha=None, is_selfplay=0):
        self.pv_fn = policy_value_fn
        self.mcts = MCTS_AZ(policy_value_fn, c_puct, n_playout, alpha)
        self.is_selfplay = is_selfplay
        self.n_actions = policy_value_fn.n_actions
    
    def to(self, device='cpu'):
        self.pv_fn.to(device)

    def train(self):
        self.mcts.train()

    def eval(self):
        self.mcts.eval()

    def get_action(self, env, temp=0):
        action_probs = np.zeros((self.n_actions,), dtype=np.float32)
        actions, visits = self.mcts.get_action_visits(env)
        visit_dist = softmax(np.log(visits) / max(temp, 1e-8))
        action_probs[list(actions)] = visit_dist
        if temp == 0:
            probs = np.zeros((len(visits),), dtype=np.float32)
            probs[np.where(np.array(visits) == max(visits))] = 1 / list(visits).count(max(visits))
        else:
            probs = visit_dist
        action = np.random.choice(actions, p=probs)
        if self.is_selfplay:
            self.mcts.prune_root(action)
        else:
            self.reset_player()
        return action, action_probs
