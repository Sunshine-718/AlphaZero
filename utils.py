#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 21/Jul/2024  03:52
import torch
import torch.nn.functional as F
import numpy as np
from numba import jit


@jit(nopython=True)
def check_winner(board):
    # Dimensions of the board
    rows, cols = board.shape

    # Check horizontal locations for a win
    for row in range(rows):
        for col in range(cols - 3):
            if abs(board[row, col] + board[row, col + 1] + board[row, col + 2] + board[row, col + 3]) == 4:
                return board[row, col]

    # Check vertical locations for a win
    for row in range(rows - 3):
        for col in range(cols):
            if abs(board[row, col] + board[row + 1, col] + board[row + 2, col] + board[row + 3, col]) == 4:
                return board[row, col]

    # Check positively sloped diagonals
    for row in range(rows - 3):
        for col in range(cols - 3):
            if abs(board[row, col] + board[row + 1, col + 1] + board[row + 2, col + 2] + board[row + 3, col + 3]) == 4:
                return board[row, col]

    # Check negatively sloped diagonals
    for row in range(3, rows):
        for col in range(cols - 3):
            if abs(board[row, col] + board[row - 1, col + 1] + board[row - 2, col + 2] + board[row - 3, col + 3]) == 4:
                return board[row, col]

    # If no winner, return 0
    return 0


@jit(nopython=True)
def valid_move(board):
    return [i for i in range(board.shape[1]) if 0 in board[:, i]]


@jit(nopython=True)
def valid_bin(board):
    return np.array([1 if 0 in board[:, i] else 0 for i in range(board.shape[1])])


@jit(nopython=True)
def place(board, action, turn):
    if action in valid_move(board):
        row_index = max(np.where(board[:, action] == 0)[0])
        board[row_index, action] = turn
        return True
    return False


@jit(nopython=True)
def check_draw(board):
    return len(np.where(board == 0)[0]) == 0


def step(board, action, turn):
    if place(board, action, turn):
        winner = check_winner(board)
        if check_draw(board):
            return board, 0, True, True
        elif winner != 0:
            if winner == turn:
                return board, 1, True, True
            return board, -1, True, True
        else:
            return board, 0, False, True
    return board, 0, False, False


@jit(nopython=True)
def board_to_state(board, turn):
    temp = np.zeros((1, 3, board.shape[0], board.shape[1]), dtype=np.float32)
    temp[:, 0] = board == 1
    temp[:, 1] = board == -1
    if turn == 1:
        temp[:, 2] = np.zeros(
            (board.shape[0], board.shape[1]), dtype=np.float32)
    else:
        temp[:, 2] = np.ones(
            (board.shape[0], board.shape[1]), dtype=np.float32)
    return temp


def print_row(action, probX, probO, max_X, max_O):
    print('⭐️ ' if probX == max_X else '   ', end='')
    print(f'action: {action}, prob_X: {probX * 100: 02.2f}%', end='\t')
    print('⭐️ ' if probO == max_O else '   ', end='')
    print(f'action: {action}, prob_O: {probO * 100: 02.2f}%')


def inspect(net, board=None):
    if board is None:
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

    pass