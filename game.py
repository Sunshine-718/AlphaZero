#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:55
import numpy as np


class Game:
    def __init__(self, env):
        self.env = env

    def start_play(self, player1, player2, discount=1, show=1):
        self.env.reset()
        players = [None, player1, player2]
        if show:
            self.env.show()
        while True:
            current_turn = self.env.turn
            player = players[current_turn]
            action = player.get_action(self.env, discount=discount)
            self.env.step(action[0])
            if show:
                self.env.show()
            if self.env.done():
                winner = self.env.winPlayer()
                if show:
                    if winner != 0:
                        print('Game end. Winner is', [
                              None, 'X', 'O'][int(winner)])
                    else:
                        print('Game end. Draw')
                return winner
    
    def single_player(self, player, temp=1, first_n_steps=12, discount=0.99, dirichlet_alpha=0.3, max_reward=None):
        assert(max_reward is not None)
        state, _ = self.env.reset()
        states, mcts_probs, rewards, next_states = [], [], [], []
        steps = 0
        while True:
            if steps < first_n_steps:
                action, probs = player.get_action(
                    self.env, temp, dirichlet_alpha, discount)
            else:
                action, probs = player.get_action(
                    self.env, 1e-3, dirichlet_alpha, discount)
            steps += 1
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            if terminated:
                reward = -10
            states.append(state)
            mcts_probs.append(probs)
            rewards.append(reward)
            next_states.append(next_state)
            if terminated or truncated:
                discounts = np.array(list([pow(discount, i) for i in range(steps)]))
                rewards = np.array(rewards) / max_reward
                assert(len(discounts) == len(rewards))
                reward = np.sum(discounts * rewards)
                winner_z = [reward for _ in range(steps)]
                return None, zip(states, mcts_probs, winner_z, next_states)
            state = next_state
            

    def start_self_play(self, player, temp=1, first_n_steps=5, show=0, discount=0.99, dirichlet_alpha=0.3):
        self.env.reset()
        states, mcts_probs, current_players, next_states = [], [], [], []
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
                        winner_z[idx] = i * \
                            pow(discount, len(winner_z) - idx - 1)
                if show:
                    if winner != 0:
                        print(f"Game end. Wineer is Player: {
                              [None, 'X', 'O'][int(winner)]}")
                    else:
                        print('Game end. Draw')
                return winner, zip(states, mcts_probs, winner_z, next_states)
