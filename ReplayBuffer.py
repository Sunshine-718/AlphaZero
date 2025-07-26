#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    def __init__(self, state_dim, capacity, action_dim, row, col, replay_ratio=0.1, device='cpu'):
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
        self.n = torch.full_like(self.value, 0, dtype=torch.int64, device=device)
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
        self.n = torch.full_like(self.n, 0, dtype=torch.int64)
        self.count = 0

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.value = self.value.to(device)
        self.next_state = self.next_state.to(device)
        self.done = self.done.to(device)
        self.n = self.n.to(device)
        self.device = device

    def store(self, state, prob, value, next_state, done, n):
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
        self.n[idx] = n
        return idx
    
    def get(self, indices):
        return self.state[indices], self.prob[indices], self.value[indices], \
            self.next_state[indices], self.done[indices], self.n[indices]
    
    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.get(idx)

    def dataloader(self, batch_size):
        total = self.__len__()
        if self.__len__() > 1000:
            total = int(self.__len__() * self.replay_ratio)
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), total, dtype=np.int64))
        dataset = TensorDataset(*self.get(idx))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def sample_balanced(self, batch_size):
        values = self.value[:self.__len__()].squeeze()  # [N]
        unique_vals = torch.unique(values)
        n_types = unique_vals.numel()
        assert(n_types <= 3)
        n_per_type = batch_size // n_types
        remainder = batch_size % n_types

        indices = []
        for i, val in enumerate(unique_vals):
            idxs = (values == val).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            size = n_per_type + (1 if i < remainder else 0)
            choice = idxs[torch.randint(0, len(idxs), (size,))]
            indices.append(choice)
        if len(indices) == 0:
            raise ValueError("No available data to sample.")
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        return self.get(indices)
