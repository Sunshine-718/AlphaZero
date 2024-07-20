#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 13/Jul/2024  23:13
from env import Env
from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from Network import Network


def symmetric_state(state, prob):
    state = deepcopy(state)
    for idx, i in enumerate(state[0]):
        state[0, idx] = np.fliplr(i)
    return state, prob[::-1]


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
