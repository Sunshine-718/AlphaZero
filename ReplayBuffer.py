#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, row, col, replay_ratio=0.25, device='cpu'):
        self.state = torch.full(
            (capacity, state_dim, row, col), torch.nan, dtype=torch.float32, device=device)
        self.prob = torch.full(
            (capacity, action_dim), torch.nan, dtype=torch.float32, device=device)
        self.value = torch.full((capacity, 1), torch.nan,
                                dtype=torch.float32, device=device)
        self.next_state = torch.full_like(
            self.state, torch.nan, dtype=torch.float32, device=device)
        self.done = torch.full_like(
            self.value, torch.nan, dtype=torch.bool, device=device)
        self.mask = torch.full((capacity, action_dim), torch.nan, dtype=torch.bool, device=device)
        self.replay_ratio = replay_ratio
        self.count = 0
        self.device = device

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
        self.next_state = torch.full_like(
            self.next_state, torch.nan, dtype=torch.float32)
        self.done = torch.full_like(self.done, torch.nan, dtype=torch.bool)
        self.mask = torch.full_like(self.mask, torch.nan, dtype=torch.bool)
        self.count = 0

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.value = self.value.to(device)
        self.next_state = self.next_state.to(device)
        self.done = self.done.to(device)
        self.mask = self.mask.to(device)
        self.device = device

    def store(self, state, prob, value, next_state, done, mask):
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
        self.done[idx] = done
        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        self.mask[idx] = mask
        return idx
    
    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.state[idx], self.prob[idx, :], self.value[idx, :], self.next_state[idx], self.done[idx], self.mask[idx]

    def dataloader(self, batch_size):
        total = self.__len__()
        if self.__len__() > 1000:
            total = int(self.__len__() * self.replay_ratio)
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), total, dtype=np.int64))
        dataset = TensorDataset(self.state[idx],
                                self.prob[idx],
                                self.value[idx],
                                self.next_state[idx],
                                self.done[idx],
                                self.mask[idx])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
