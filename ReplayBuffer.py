#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim):
        self.state = torch.full(
            (capacity, state_dim, 6, 7), torch.nan, dtype=torch.float32)
        self.prob = torch.full(
            (capacity, action_dim), torch.nan, dtype=torch.float32)
        self.value = torch.full((capacity, 1), torch.nan, dtype=torch.float32)
        self.count = 0
        self.device = 'cpu'

    def __len__(self):
        return min(self.count, len(self.state))

    def reset(self):
        self.state = torch.full_like(
            self.state, torch.nan, dtype=torch.float32)
        self.prob = torch.full_like(
            self.prob, torch.nan, dtype=torch.float32)
        self.value = torch.full_like(
            self.value, torch.nan, dtype=torch.float32)
        self.count = 0

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.value = self.value.to(device)
        self.device = device

    def store(self, state, prob, value):
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
        return idx

    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.state[idx], self.prob[idx, :], self.value[idx, :]

    def dataloader(self, batch_size):
        dataset = TensorDataset(self.state[:self.__len__(
        )], self.prob[:self.__len__()], self.value[:self.__len__()])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class ReplayBufferQ(ReplayBuffer):
    def __init__(self, state_dim, capacity, action_dim):
        super().__init__(state_dim, capacity, action_dim)
        self.action = torch.full((capacity, 1), torch.nan, dtype=torch.float32)

    def reset(self):
        super().reset()
        self.action = torch.full_like(
            self.action, torch.nan, dtype=torch.float32)

    def to(self, device='cpu'):
        super().to(device)
        self.action.to(device)

    def store(self, state, action, prob, value):
        idx = super().store(state, prob, value)
        self.action[idx] = action

    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.state[idx], self.action[idx], self.prob[idx, :], self.value[idx, :]
    
    def dataloader(self, batch_size):
        dataset = TensorDataset(self.state[:self.__len__(
        )], self.action[:self.__len__()], self.prob[:self.__len__()], self.value[:self.__len__()])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
