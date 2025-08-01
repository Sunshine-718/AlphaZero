#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:55
import numpy as np


class Game:
    def __init__(self, env):
        self.env = env

    def start_play(self, player1, player2, show=1):
        self.env.reset()
        players = [None, player1, player2]
        if show:
            self.env.show()
        while True:
            current_turn = self.env.turn
            player = players[current_turn]
            action, probs, *_ = player.get_action(self.env)
            player.reset_player()
            prev_env = self.env.copy()
            self.env.step(action)
            if show:
                self.env.show()
            if self.env.done():
                winner = self.env.winPlayer()
                if show:
                    if winner != 0:
                        print('Game end. Winner is', (None, 'X', 'O')[int(winner)])
                    else:
                        print('Game end. Draw')
                return winner

    def start_self_play(self, player, temp=1, first_n_steps=5):
        self.env.reset()
        states, mcts_probs, current_players, next_states = [], [], [], []
        # values = []
        steps = 0
        while True:
            temperature = 1e-3 if steps >= first_n_steps else temp
            action, probs = player.get_action(self.env, temperature)
            # values.append(float(v_target))
            steps += 1
            states.append(self.env.current_state())
            mcts_probs.append(probs)
            current_players.append(self.env.turn)
            self.env.step(action)
            next_states.append(self.env.current_state())
            if self.env.done():
                winner = self.env.winPlayer()
                winner_z = np.zeros(len(current_players))
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1
                    winner_z[np.array(current_players) != winner] = -1
                dones = [False]*len(current_players)
                dones[-1] = True
                return winner, zip(states, mcts_probs, winner_z, winner_z, next_states, dones)
