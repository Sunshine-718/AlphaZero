#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47
import numpy as np
from MCTS import MCTS, MCTS_AZ
from abc import abstractmethod, ABC
from utils import softmax, policy_value_fn


class Player(ABC):
    def __init__(self):
        self.win_rate = float('nan')
        self.mcts = None

    def reset_player(self):
        raise NotImplementedError

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
        self.deterministic = deterministic

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def get_action(self, env, discount, *, compute_winrate=False):
        action_probs, value = self.net(env)
        if compute_winrate:
            self.win_rate = (value + 1) / 2
        if self.deterministic:
            return max(action_probs, key=lambda x: x[1])[0], None
        else:
            actions, probs = list(zip(*action_probs))
            return np.random.choice(actions, p=softmax(probs)), None


class Human(Player):
    def get_action(self, *args, **kwargs):
        move = int(input('Your move: '))
        return move, None


class MCTSPlayer(Player):
    def __init__(self, c_puct=4, n_playout=2000, num_worker=1):
        super().__init__()
        if num_worker == 1:
            self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, discount=1, *, compute_winrate=False):
        valid = env.valid_move()
        if len(valid) > 0:
            action = self.mcts.get_action(env, discount)
            if compute_winrate:
                Q = self.mcts.root.children[action].Q
                self.win_rate = (Q + 1) / 2
            self.reset_player()
            return action
        else:
            print('Warning: the board is full')


class AlphaZeroPlayer(Player):
    def __init__(self, policy_value_fn, c_puct=4, n_playout=1000, is_selfplay=0):
        super().__init__()
        self.mcts = MCTS_AZ(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def train(self):
        self.mcts.train()

    def eval(self):
        self.mcts.eval()

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=0, dirichlet_alpha=0.3, discount=1, *, compute_winrate=False):
        valid = env.valid_move()
        action_probs = np.zeros((7,), dtype=np.float32)
        if len(valid) > 0:
            actions, visits = self.mcts.get_action_visits(
                env, dirichlet_alpha, discount)
            if temp == 0:
                probs = np.zeros((len(visits),), dtype=np.float32)
                probs[np.where(np.array(visits) == max(visits))
                      ] = 1 / list(visits).count(max(visits))
            else:
                probs = softmax(np.log(np.array(visits) + 1e-8) / temp)
            action = np.random.choice(actions, p=probs)
            action_probs[list(actions)] = probs
            if compute_winrate:
                Q = self.mcts.root.children[action].Q
                self.win_rate = (Q + 1) / 2
            if self.is_selfplay:
                self.mcts.update_with_move(action)
            else:
                self.reset_player()
            return action, action_probs
        else:
            print('WARNING: the board is full')
