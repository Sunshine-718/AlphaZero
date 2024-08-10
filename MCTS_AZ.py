#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  19:36
import numpy as np
from MCTS import MCTS
from copy import deepcopy


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