#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:59
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ReplayBuffer:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, state_dim, capacity, action_dim, row, col, replay_ratio=0.1, device='cpu', balance_done_value=True):
        if not self._initialized:
            self.state = torch.full(
                (capacity, state_dim, row, col), torch.nan, dtype=torch.float32, device=device)
            self.prob = torch.full(
                (capacity, action_dim), torch.nan, dtype=torch.float32, device=device)
            self.value = torch.full((capacity, 1), torch.nan,
                                    dtype=torch.float32, device=device)
            self.winner = torch.full(
                (capacity, 1), 0, dtype=torch.int32, device=device)
            self.next_state = torch.full_like(
                self.state, torch.nan, dtype=torch.float32, device=device)
            self.done = torch.full_like(
                self.value, torch.nan, dtype=torch.bool, device=device)
            self.replay_ratio = replay_ratio
            self.device = device
            self.balance_done_value = balance_done_value
            self.current_capacity = 2500
            self._ptr = 0

    def __len__(self):
        return len(self.value[~self.value.isnan()])
    
    def double(self):
        if self.current_capacity < len(self.state):
            self.current_capacity = min(len(self.state), self.current_capacity * 2)
            self._ptr = len(self) % len(self.state)
            assert(0 <= self._ptr < self.current_capacity)

    def is_full(self):
        return self.__len__() >= len(self.state)

    def reset(self):
        self.state = torch.full_like(self.state, torch.nan, dtype=torch.float32)
        self.prob = torch.full_like(self.prob, torch.nan, dtype=torch.float32)
        self.value = torch.full_like(self.value, torch.nan, dtype=torch.float32)
        self.winner = torch.full_like(self.winner, torch.nan, dtype=torch.int32)
        self.next_state = torch.full_like(self.next_state, torch.nan, dtype=torch.float32)
        self.done = torch.full_like(self.done, torch.nan, dtype=torch.bool)
        self._ptr = 0
        self.current_capacity = 2500

    def to(self, device='cpu'):
        self.state = self.state.to(device)
        self.prob = self.prob.to(device)
        self.value = self.value.to(device)
        self.winner = self.winner.to(device)
        self.next_state = self.next_state.to(device)
        self.done = self.done.to(device)
        self.device = device

    def store(self, state, prob, value, winner, next_state, done):
        idx = self._ptr
        self._ptr = (self._ptr + 1) % self.current_capacity
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        self.state[idx] = state
        if isinstance(prob, np.ndarray):
            prob = torch.from_numpy(prob).float().to(self.device)
        self.prob[idx] = prob
        self.value[idx] = value
        self.winner[idx] = winner
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float().to(self.device)
        self.next_state[idx] = next_state
        self.done[idx] = done
        return idx

    def get(self, indices):
        return self.state[indices], self.prob[indices], self.value[indices], \
            self.winner[indices], self.next_state[indices], self.done[indices]

    def sample(self, batch_size):
        idx = torch.from_numpy(np.random.randint(
            0, self.__len__(), batch_size, dtype=np.int64))
        return self.get(idx)

    def dataloader(self, batch_size):
        total_samples = self.__len__()
        max_samples = min(
            int(total_samples * self.replay_ratio), 10000, total_samples)
        if total_samples <= 1000:
            max_samples = total_samples
        if max_samples <= 0:
            raise ValueError("No available data to sample.")

        done_flags = self.done[:total_samples].squeeze()
        winner_labels = self.winner[:total_samples].squeeze()
        done_indices = (done_flags == 1).nonzero(as_tuple=True)[0]
        not_done_indices = (done_flags == 0).nonzero(as_tuple=True)[0]

        n_done = max(1, int(max_samples * 0.2))
        n_not_done = max_samples - n_done

        def balanced_or_random_sample(indices, winner, n, do_balance):
            if do_balance and len(indices) > 0 and torch.allclose(winner[indices], winner[indices].round()):
                labels = winner[indices].long()
                unique_vals = torch.unique(labels)
                n_types = unique_vals.numel()
                n_per_type = n // n_types
                remainder = n % n_types
                result = []
                for i, val in enumerate(unique_vals):
                    idxs = indices[(labels == val).nonzero(as_tuple=True)[0]]
                    size = n_per_type + (1 if i < remainder else 0)
                    size = min(size, len(idxs))
                    if size > 0:
                        select = idxs[torch.randperm(len(idxs))[:size]]
                        result.append(select)
                if result:
                    return torch.cat(result)
                else:
                    return torch.tensor([], dtype=torch.long)
            else:
                if len(indices) == 0:
                    return torch.tensor([], dtype=torch.long)
                size = min(len(indices), n)
                return indices[torch.randperm(len(indices))[:size]]

        done_sample_idx = balanced_or_random_sample(
            done_indices, winner_labels, n_done, self.balance_done_value)

        if len(not_done_indices) == 0:
            not_done_sample_idx = torch.tensor([], dtype=torch.long)
        else:
            size = min(len(not_done_indices), n_not_done)
            not_done_sample_idx = not_done_indices[torch.randperm(len(not_done_indices))[:size]]

        idx = torch.cat([done_sample_idx, not_done_sample_idx])
        if len(idx) == 0:
            raise ValueError("No available data to sample.")
        idx = idx[torch.randperm(len(idx))]  # 总体再打乱

        dataset = TensorDataset(*self.get(idx))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def sample_balanced(self, batch_size):
        # 采样上限逻辑
        current_len = self.__len__()
        sample_size = min(current_len, min(int(len(self.state) * self.replay_ratio), 10000))

        winner = self.winner[:current_len].squeeze()  # [N]
        if not torch.allclose(winner, winner.round()):
            raise ValueError("ReplayBuffer的value数据非整数，平衡采样前需保证离散标签！")
        winner = winner.long()
        unique_vals = torch.unique(winner)
        n_types = unique_vals.numel()
        assert (n_types <= 3)
        total = sample_size
        n_per_type = total // n_types
        remainder = total % n_types

        indices = []
        for i, val in enumerate(unique_vals):
            idxs = (winner == val).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                continue
            size = min(len(idxs), n_per_type + (1 if i < remainder else 0))
            if size == 0:
                continue
            choice = idxs[torch.randint(0, len(idxs), (size,))]
            indices.append(choice)
        if len(indices) == 0:
            raise ValueError("No available data to sample.")
        indices = torch.cat(indices)
        indices = indices[torch.randperm(len(indices))]
        dataset = TensorDataset(*self.get(indices))
        actual_batch = min(len(indices), batch_size)
        return DataLoader(dataset, batch_size=actual_batch, shuffle=True)
