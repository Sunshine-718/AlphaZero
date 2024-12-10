#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, row, col):
        self.state = torch.full(
            (capacity, state_dim, row, col), torch.nan, dtype=torch.float32)
        self.prob = torch.full(
            (capacity, action_dim), torch.nan, dtype=torch.float32)
        self.value = torch.full((capacity, 1), torch.nan, dtype=torch.float32)
        self.next_state = torch.full_like(self.state, torch.nan, dtype=torch.float32)
        self.count = 0
        self.device = 'cpu'

    def __len__(self):
        return min(self.count, len(self.state))

    def is_full(self):
        return self.__len__() >= len(self.state)

    def reset(self):
        self.state = torch.full_like(
            self.state, torch.nan, dtype=torch.float32)
        self.prob = torch.full_like(
            self.prob, torch.nan, dtype=torch.float32)
        self.value = torch.full_like(
            self.value, torch.nan, dtype=torch.float32)
        self.next_state = torch.full_like(self.next_state, torch.nan, dtype=torch.float32)
        self.count = 0

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.value = self.value.to(device)
        self.next_state = self.next_state.to(device)
        self.device = device

    def store(self, state, prob, value, next_state):
        idx = self.count % len(self.state)
        self.count += 1
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).type(
                torch.FloatTensor).to(self.device)
        self.state[idx] = state
        if isinstance(prob, np.ndarray):
            prob = torch.from_numpy(prob).type(
                torch.FloatTensor).to(self.device)
        self.prob[idx] = prob
        self.value[idx] = value
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).type(
                torch.FloatTensor).to(self.device)
        self.next_state[idx] = next_state
        return idx

    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.state[idx], self.prob[idx, :], self.value[idx, :], self.next_state[idx]

    def dataloader(self, batch_size):
        dataset = TensorDataset(self.state[:self.__len__(
        )], self.prob[:self.__len__()], self.value[:self.__len__()], self.next_state[:self.__len__()])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
