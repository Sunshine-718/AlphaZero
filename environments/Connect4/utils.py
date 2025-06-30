#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  05:58
import torch
import numpy as np
from numba import njit
from copy import deepcopy


@njit(fastmath=True)
def board_to_state(board, turn):
    temp = np.zeros((1, 3, board.shape[0], board.shape[1]), dtype=np.float32)
    temp[:, 0] = board == 1
    temp[:, 1] = board == -1
    if turn == 1:
        temp[:, 2] = np.ones((board.shape[0], board.shape[1]), dtype=np.float32)
    else:
        temp[:, 2] = -np.ones((board.shape[0], board.shape[1]), dtype=np.float32)
    return temp


@njit(fastmath=True)
def check_full(board):
    return len(np.where(board == 0)[0]) == 0


@njit(fastmath=True)
def check_winner(board):
    rows, cols = board.shape

    for row in range(rows):
        for col in range(cols - 3):
            current = board[row, col]
            if current != 0 and current == board[row, col+1] and current == board[row, col+2] and current == board[row, col+3]:
                return current

    for row in range(rows - 3):
        for col in range(cols):
            current = board[row, col]
            if current != 0 and current == board[row+1, col] and current == board[row+2, col] and current == board[row+3, col]:
                return current

    for row in range(rows - 3):
        for col in range(cols - 3):
            current = board[row, col]
            if current != 0 and current == board[row+1, col+1] and current == board[row+2, col+2] and current == board[row+3, col+3]:
                return current

    for row in range(3, rows):
        for col in range(cols - 3):
            current = board[row, col]
            if current != 0 and current == board[row-1, col+1] and current == board[row-2, col+2] and current == board[row-3, col+3]:
                return current

    return 0


def inspect(net, board=None):
    if board is None:
        board = np.array([[0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0]])
    with torch.no_grad():
        state0 = torch.from_numpy(board_to_state(
            board, 1)).float().to(net.device)
        mask0 = torch.tensor(valid_mask(board), dtype=torch.bool, device=net.device).unsqueeze(0)
        probs0, value0 = net.policy_value(state0, mask0)
        probs0, value0 = probs0.flatten(), float(value0[0, 0])
        board[5, 3] = 1
        state1 = torch.from_numpy(board_to_state(
            board, -1)).float().to(net.device)
        mask1 = torch.tensor(valid_mask(board), dtype=torch.bool, device=net.device).unsqueeze(0)
        probs1, value1 = net.policy_value(state1, mask1)
        probs1, value1 = probs1.flatten(), float(value1[0, 0])
    for (idx, pX), (_, pO) in zip(enumerate(probs0), enumerate(probs1)):
        print_row(idx, pX, pO, np.max(probs0), np.max(probs1))
    print(f'State-value X: {value0: .4f}\nState-value O: {value1: .4f}')
    return probs0, value0, probs1, value1


def instant_augment(batch):
    state, prob, value, next_state, done, mask = batch

    state_flipped = torch.flip(state, dims=[3])
    next_state_flipped = torch.flip(next_state, dims=[3])
    prob_flipped = torch.flip(prob, dims=[1])
    mask_flipped = torch.flip(mask, dims=[1])

    state = torch.cat([state, state_flipped], dim=0)
    next_state = torch.cat([next_state, next_state_flipped], dim=0)
    prob = torch.cat([prob, prob_flipped], dim=0)
    mask = torch.cat([mask, mask_flipped], dim=0)
    value = torch.cat([value, value], dim=0)
    done = torch.cat([done, done], dim=0)

    return state, prob, value, next_state, done, mask


@njit
def place(board, action, turn):
    if action in valid_move(board):
        row_index = max(np.where(board[:, action] == 0)[0])
        board[row_index, action] = turn
        return True
    return False


def print_row(action, probX, probO, max_X, max_O):
    print('⭐️ ' if probX == max_X else '   ', end='')
    print(f'action: {action}, prob_X: {probX * 100: 02.2f}%', end='\t')
    print('⭐️ ' if probO == max_O else '   ', end='')
    print(f'action: {action}, prob_O: {probO * 100: 02.2f}%')


@njit
def valid_move(board):
    return [i for i in range(board.shape[1]) if 0 in board[:, i]]

@njit
def valid_mask(board):
    return [0 in board[:, i] for i in range(board.shape[1])]
