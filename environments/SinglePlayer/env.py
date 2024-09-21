#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 19/Sep/2024  06:37
import gymnasium as gym


class Env:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.state = None
        self._done = False
        self.n_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.high.shape[0]
        self.reward = 0

    def reset(self):
        self.state, _ = self.env.reset()
        self._done = False
        return self.state, None

    def done(self):
        return self._done

    def valid_move(self):
        return list(range(self.env.action_space.n))

    def current_state(self):
        return self.state.reshape(1, -1)

    def step(self, action):
        self.state, self.reward, terminated, truncated, _ = self.env.step(action)
        self._done = terminated or truncated
        return self.state, self.reward, terminated, truncated, None
