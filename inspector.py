#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 13/Jul/2024  23:13
from env import Env, board_to_state
from copy import deepcopy
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from copy import deepcopy
from Network import Network


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


def inspect(net):
    board = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]])
    state0 = torch.from_numpy(board_to_state(
        board, 0)).float().cuda()
    p0, v0 = net(state0)
    probs0 = F.softmax(p0, dim=1).detach().cpu().numpy().flatten()
    value0 = v0.item()
    state1 = torch.from_numpy(board_to_state(
        board, 1)).float().cuda()
    p1, v1 = net(state1)
    probs1 = F.softmax(p1, dim=1).detach().cpu().numpy().flatten()
    value1 = v1.item()
    for (idx, pX), (_, pO) in zip(enumerate(probs0), enumerate(probs1)):
        print_row(idx, pX, pO, np.max(probs0), np.max(probs1))
    print(f'State-value X: {value0: .4f}, State-value O: {value1: .4f}')


if __name__ == '__main__':
    env = Env()
    net = Network(1e-3, 3, 32, 7, 'cuda')
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

    # state = board_to_state(env.board, 1)
    # print(symmetric_state(state, [i for i in range(7)]))
    pass
