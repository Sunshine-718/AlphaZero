#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:55
import numpy as np


class Game:
    def __init__(self, env):
        self.env = env

    def start_play(self, player1, player2, discount=1, show=1, show_nn=0):
        self.env.reset()
        players = [None, player1, player2]
        if show:
            self.env.show()
        while True:
            current_turn = self.env.turn
            player = players[current_turn]
            action, probs, *_ = player.get_action(self.env, discount=discount)
            prev_env = self.env.copy()
            self.env.step(action)
            if show:
                self.env.show()
                if show_nn:
                    try:
                        # 显示 action_probs
                        if probs is not None:
                            print("Action probabilities:")
                            for idx, p in enumerate(probs):
                                if idx in prev_env.valid_move():
                                    print(f"  Action {idx}: {p * 100:5.2f}%")

                        # 显示 value
                        if hasattr(player, 'net') and hasattr(player.net, 'policy_value_fn'):
                            _, value = player.net.policy_value_fn(prev_env)
                            print(f'Estimated value (win prob): {value:+.4f}')
                        elif hasattr(player, 'mcts'):
                            _, value = player.mcts.policy(prev_env)
                            print(f'Estimated value (win prob): {value:+.4f}')
                    except Exception as e:
                        pass
            if self.env.done():
                winner = self.env.winPlayer()
                if show:
                    if winner != 0:
                        print('Game end. Winner is', [None, 'X', 'O'][int(winner)])
                    else:
                        print('Game end. Draw')
                return winner

    def start_self_play(self, player, temp=1, first_n_steps=5, show=0, discount=0.99, dirichlet_alpha=0.3):
        self.env.reset()
        states, mcts_probs, current_players, next_states, masks = [], [], [], [], []
        # values = []
        steps = 0
        while True:
            if steps < first_n_steps:
                action, probs = player.get_action(
                    self.env, temp, dirichlet_alpha, discount)
            else:
                action, probs = player.get_action(
                    self.env, 1e-3, dirichlet_alpha, discount)
            steps += 1
            states.append(self.env.current_state())
            masks.append(self.env.valid_mask())
            node = player.mcts.root
            if not node.children:
                v = node.Q
            else:
                pair = [(child.n_visits, -child.Q) for child in node.children.values()]
                visits, Q = zip(*pair)
                weights = np.array([i / sum(visits) for i in visits])
                Q = np.array(Q)
                v = np.sum(weights * Q)
            values.append(v)
            mcts_probs.append(probs)
            current_players.append(self.env.turn)
            self.env.step(action)
            next_states.append(self.env.current_state())
            if show:
                self.env.show()
            if self.env.done():
                winner = self.env.winPlayer()
                winner_z = np.zeros(len(current_players))
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1
                    winner_z[np.array(current_players) != winner] = -1
                    for idx, i in enumerate(winner_z):
                        winner_z[idx] = i * pow(discount, len(winner_z) - idx - 1)
                if show:
                    if winner != 0:
                        print(f"Game end. Wineer is Player: {[None, 'X', 'O'][int(winner)]}")
                    else:
                        print('Game end. Draw')
                dones = [False for _ in range(len(current_players))]
                dones[-1] = True
                # return winner, zip(states, mcts_probs, winner_z, next_states, dones, masks)
                values = np.array(values)
                winner_z = np.array(winner_z)
                ratio = 0.25
                values = ratio * values + (1 - ratio) * winner_z
                return winner, zip(states, mcts_probs, values, next_states, dones, masks)
