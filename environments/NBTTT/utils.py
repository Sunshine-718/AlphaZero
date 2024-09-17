# -*- coding: utf-8 -*-
# @Time: 2024/4/23 5:08
from ctypes import cdll, c_char_p, c_int, POINTER
import numpy as np
import time
from numba import njit
from copy import deepcopy
import torch
import platform

system = platform.system()
mapping = {'Windows': 'dll',
           'Darwin': 'dylib',
           'Linux': 'so'}

lib = cdll.LoadLibrary(f"./environments/NBTTT/NBTTTAgent.{mapping[system]}")


def Timer(precision=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            output = func(*args, **kwargs)
            end = time.perf_counter()
            print(f'Process Time: {round(end - start, precision)}s')
            return output

        return wrapper

    return decorator


def print_board(board):
    def print_board_row(bd, a, b, c, i, j, k):
        s = ["X", "O", '.']
        print(" " + s[bd[a][i]] + " " + s[bd[a][j]] + " " + s[bd[a][k]] + " | "
              + s[bd[b][i]] + " " + s[bd[b][j]] + " " + s[bd[b][k]] + " | "
              + s[bd[c][i]] + " " + s[bd[c][j]] + " " + s[bd[c][k]])

    print_board_row(board, 1, 2, 3, 1, 2, 3)
    print_board_row(board, 1, 2, 3, 4, 5, 6)
    print_board_row(board, 1, 2, 3, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 4, 5, 6, 1, 2, 3)
    print_board_row(board, 4, 5, 6, 4, 5, 6)
    print_board_row(board, 4, 5, 6, 7, 8, 9)
    print(" ------+-------+------")
    print_board_row(board, 7, 8, 9, 1, 2, 3)
    print_board_row(board, 7, 8, 9, 4, 5, 6)
    print_board_row(board, 7, 8, 9, 7, 8, 9)
    print()


class ndarray:
    def __init__(self, array, shape):
        self.array = array
        self.shape = shape

    def __len__(self):
        temp = 1
        for i in self.shape:
            temp *= i
        return temp

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.array[index]
        return self.__class__(self.array[index[0]], self.shape[1:])[index[1:]]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self.array[index] = value
        elif isinstance(index, tuple):
            if len(index) == 1:
                self.array[index[0]] = value
            else:
                self.__class__(self.array[index[0]], self.shape[1:]).__setitem__(
                    index[1:], value)
        else:
            raise TypeError("Invalid argument type.")

    def numpy(self):
        if len(self.shape) > 2:
            raise NotImplementedError
        if len(self.shape) == 1:
            return np.array([self.array[int(i)] for i in range(self.shape[0])], dtype=np.int8)
        else:
            return np.array([self.array[int(i)][int(j)] for i in range(self.shape[0]) for j in range(self.shape[1])],
                            dtype=np.int8).reshape(*self.shape)


def c_string(string: str):
    return c_char_p(b'%s' % string.encode())


def hello(name: c_char_p):
    assert (isinstance(name, c_char_p))
    lib.hello(name)


def create_1d_array_int(array):
    return ndarray((c_int * len(array))(*array), [len(array)])


def freeArray(array):
    lib.freeArray(array.array)
    del array


def print_1d_array_int(array):
    return lib.print_array(array.array, len(array))


def print_2d_array_int(array: ndarray):
    lib.print_2d_array(array.array, array.shape[0], array.shape[1])


def zeros_2d_int(row, col):
    lib.zeros.argtypes = [c_int, c_int]
    lib.zeros.restype = POINTER(POINTER(c_int))
    return ndarray(lib.zeros(row, col), [row, col])


def free2dArray(array: ndarray):
    lib.freeMatrix(array.array, array.shape[0])
    del array


def alphabeta(boards, actions, m, curr, depth, branch, player):
    lib.alphabeta.argtypes = [POINTER(c_int), POINTER(
        c_int), c_int, c_int, c_int, c_int, c_int]
    lib.alphabeta.restype = c_int
    return lib.alphabeta(boards.array, actions.array, m, curr, depth, branch, player)


@njit
def count_pieces(board) -> int:
    """
    Function to count total pieces in board
    :param board: game board
    :return: number of pieces
    """
    return len(np.where(board != 2)[0])


def dynamic_params(boards: np.ndarray, curr: int, depth: int, branch: int) -> (int, int):
    """
    Function to provide dynamic depth and branch.
    :param boards: nine boards.
    :param curr: index of current board.
    :param depth: maximum search depth.
    :param branch: maximum node to expand at each level of tree, default 9 (maximum number of action in one board).
    :return: depth, branch.
    """
    step = count_pieces(boards)
    num_pieces = list(sorted([count_pieces(i) for i in boards]))
    steps = [0, 10, 15, 22, 27, 31, 35]
    if steps[0] < step < steps[1]:
        return 9, 9
    elif steps[1] <= step < steps[2]:
        depth, branch = 11, 9
    elif steps[2] <= step < steps[3]:
        depth, branch = 13, 9
    elif steps[3] <= step < steps[4]:
        depth, branch = 22, 9
    elif steps[4] <= step < steps[5]:
        depth, branch = 24, 9
    elif steps[5] <= step < steps[6]:
        depth, branch = 26, 9
    elif step >= steps[6]:
        depth, branch = 81, 9
    if 12 <= step <= 25 and num_pieces[0] <= 1:
        depth = max(depth - 1, 6)
        branch = max(branch - 1, 5)
    return depth, branch


@njit
def check_winner_single_board(board: np.ndarray) -> int:
    """
    Function to check winner in one sub-board.
    :param board: sub-board, shape(3, 3).
    :return: 0, 1 or -1. 1 means I win, -1 means opponent wins, 0 means draw or game not end yet.
    """
    for row in board:
        if np.all(row == 1):
            return 1
        elif np.all(row == -1):
            return -1
    for col in board.T:
        if np.all(col == 1):
            return 1
        elif np.all(col == -1):
            return -1
    diag1 = np.diag(board)
    diag2 = np.diag(np.fliplr(board))
    if np.all(diag1 == 1) or np.all(diag2 == 1):
        return 1
    elif np.all(diag1 == -1) or np.all(diag2 == -1):
        return -1
    return 0


@njit
def check_draw(boards: np.ndarray, curr: int) -> bool:
    """
    Function to check whether the game is draw.
    :param boards: nine board, shape(10, 10)
    :param curr: index of current board.
    :return: True or False.
    """
    return len(np.where(boards[curr, 1:] != 2)[0]) == 9


@njit
def get_board(boards: np.ndarray, index: int, view: int) -> np.ndarray:
    """
    Function to get a sub-board from nine board.
    :param boards: nine board, shape(10, 10).
    :param index: index of row.
    :param view: from my view or opponent's view, 0 (me) and 1 (opponent).
    :return: sub-board with shape (3, 3), with -1, 0, 1. -1 is opponent's pieces, 0 is empty, 1 is my pieces.
    """
    if 1 <= index <= 9:
        board = boards[index, 1:].copy()
        if view == 0:
            board[board == 1] = -1
            board[board == 0] = 1
        else:
            board[board == 0] = -1
            board[board == 1] = 1
        board[board == 2] = 0
        return board.reshape(3, 3)
    else:
        raise IndexError(f"Index {index} out of range.")


@njit
def winPlayer(boards):
    for i in range(1, 10):
        board = get_board(boards, i, 0)
        winner = check_winner_single_board(board)
        if winner != 0:
            return winner
    return 0


@njit
def valid_move(boards: np.ndarray, curr: int) -> list:
    """
    Function to check the valid action in given board.
    :param boards: nine board, shape(10, 10)
    :param curr: index of current board.
    :return: list of valid action.
    """
    return list(np.where(boards[curr, 1:] == 2)[0])

def board_to_state(curr, boards, turn):
    boards = boards[1:, 1:].copy().reshape(1, 9, 3, 3)
    x_pieces = np.zeros_like(boards, dtype=np.float32)
    x_pieces[boards == 0] = 1
    o_pieces = np.zeros_like(boards, dtype=np.float32)
    o_pieces[boards == 1] = 1
    curr_dim = [np.ones((1, 1, 3, 3), dtype=np.float32) if i ==
            curr else np.zeros((1, 1, 3, 3), dtype=np.float32) for i in range(9)]
    curr_dim = np.concatenate(curr_dim, axis=1)
    turn_dim = np.zeros((1, 1, 3, 3), dtype=np.float32) + turn
    return np.concatenate([x_pieces, o_pieces, curr_dim, turn_dim], axis=1)

def swap_channel(a, mapping):
    a_copy = deepcopy(a)
    for key, value in mapping.items():
        a_copy[:, key] = a[:, value]
        a_copy[:, key + 9] = a[:, value + 9]
        a_copy[:, key + 18] = a[:, value + 18]
    return a_copy

def rot90(state, prob):
    # 0 -> 6, 6 -> 8, 8 -> 2, 2 -> 0
    # 5 -> 1, 1 -> 3, 3 -> 7, 7 -> 5
    mapping = {6: 0, 8: 6, 2: 8, 0: 2,
               1: 5, 3: 1, 7: 3, 5: 7}
    state_copy = swap_channel(state, mapping)
    prob_copy = deepcopy(prob)
    for key, value in mapping.items():
        prob_copy[:, key] = prob[:, value]
    state_copy[:, :18] = torch.rot90(state_copy[:, :18], dims=(2, 3))
    return state_copy, prob_copy

def fliplr(state, prob):
    # 0 <-> 2, 3 <-> 5, 6 <-> 8
    mapping = {0: 2, 2: 0, 3: 5,
               5: 3, 6: 8, 8: 6}
    state_copy = swap_channel(state, mapping)
    prob_copy = deepcopy(prob)
    for key, value in mapping.items():
        prob_copy[:, key] = prob[:, value]
    for idx, i in enumerate(state_copy):
        for idx_j, j in enumerate(i):
            state_copy[idx, idx_j] = torch.fliplr(j)
    return state_copy, prob_copy

def check(state):
    temp = torch.zeros((1, 9, 3, 3)) + 2
    temp[state[:, :9] == 1] = 0
    temp[state[:, 9:18] == 1] = 1
    temp = temp.reshape(9, 9)
    boards = torch.zeros((10, 10)) + 2
    boards[1:, 1:] = temp
    print_board(boards.numpy().astype(int))
    

def instant_augment(batch):
    state, prob, value, _ = deepcopy(batch)
    origin = (state, prob)
    flipped = fliplr(state, prob)
    states = []
    probs = []
    values = [value for _ in range(8)]
    for i in range(4):
        states.append(origin[0])
        states.append(flipped[0])
        probs.append(origin[1])
        probs.append(flipped[1])
        if i < 3:
            origin = rot90(*origin)
            flipped = rot90(*flipped)
    state = torch.concat(states)
    prob = torch.concat(probs)
    value = torch.concat(values)
    return state, prob, value, None

def inspect():
    raise NotImplementedError
