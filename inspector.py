#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 13/Jul/2024  23:13
from env import Env, board_to_state
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy


class Network(nn.Module):
    def __init__(self, lr, in_dim, h_dim, out_dim, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 3), padding=(2, 2)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim, h_dim * 2,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 2, h_dim * 4,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 4, h_dim * 8,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),)
        self.policy = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Flatten(),
                                    nn.Linear(h_dim * 4, h_dim * 4),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Linear(h_dim * 4, out_dim),
                                    nn.LogSoftmax(dim=1))
        self.value = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Flatten(),
                                   nn.Linear(h_dim * 4, h_dim * 4),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(h_dim * 4, 1),
                                   nn.Tanh())
        self.device = device
        self.opt = Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.weight_init()
        self.to(self.device)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is not None:
            try:
                self.load_state_dict(torch.load(path))
            except Exception as e:
                print(f'failed to load parameters\n{e}')

    def forward(self, x):
        hidden = self.hidden(x)
        return self.policy(hidden), self.value(hidden)


def symmetric_state(state, prob):
    state = deepcopy(state)
    for idx, i in enumerate(state[0]):
        state[0, idx] = np.fliplr(i)
    return state, prob[::-1]


def print_row(action, probX, probO, max_X, max_O):
    print('⭐️ ' if probX == max_X else '   ', end='')
    print(f'action: {action}, prob_X: {probX * 100: 02.2f}%', end='\t')
    print('⭐️ ' if probO == max_O else '   ', end='')
    print(f'action: {action}, prob_O: {probO * 100: 02.2f}%')
    


if __name__ == '__main__':
    env = Env()
    net = Network(1e-3, 3, 64, 7, 'cuda')
    params = './params/current.pt'
    net.load(params)
    net.eval()
    env.reset()
    env.board = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
    state0 = torch.from_numpy(board_to_state(
        env.board, 0)).float().cuda()
    p0, v0 = net(state0)
    probs0 = F.softmax(p0, dim=1).detach().cpu().numpy().flatten()
    value0 = v0.item()
    state1 = torch.from_numpy(board_to_state(
        env.board, 1)).float().cuda()
    p1, v1 = net(state1)
    probs1 = F.softmax(p1, dim=1).detach().cpu().numpy().flatten()
    value1 = v1.item()
    env.show()
    for (idx, pX), (ido, pO) in zip(enumerate(probs0), enumerate(probs1)):
        print_row(idx, pX, pO, np.max(probs0), np.max(probs1))
    print(f'State-value X: {value0: .4f}, State-value O: {value1: .4f}')
    # state = node.state
    # print(symmetric_state(state, [i for i in range(7)]))
    pass
