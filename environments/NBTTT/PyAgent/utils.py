#!/usr/bin/python3
import numpy as np
from copy import deepcopy
import time


def Timer(precision=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            output = func(*args, **kwargs)
            end = time.perf_counter()
            print(f'Process Time: {round(end - start, precision)}s')
            print('=' * 50)
            print()
            return output

        return wrapper

    return decorator


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
    steps = [0, 10, 15, 25, 30, 37, 40]
    if steps[0] <= step < steps[1]:
        return 5, 9
    elif steps[1] <= step < steps[2]:
        depth, branch = 6, 9
    elif steps[2] <= step < steps[3]:
        depth, branch = 8, 9
    elif steps[3] <= step < steps[4]:
        depth, branch = 9, 9
    elif steps[4] <= step < steps[5]:
        depth, branch = 12, 9
    elif steps[5] <= step < steps[6]:
        depth, branch = 18, 9
    elif step >= steps[6]:
        depth, branch = 81, 9
    return depth, branch


def print_board(board):
    def print_board_row(bd, a, b, c, i, j, k):
        s = ["X", "O", '.']
        print(" " + s[bd[a][i]] + " " + s[bd[a][j]] + " " + s[bd[a][k]] + " | " \
              + s[bd[b][i]] + " " + s[bd[b][j]] + " " + s[bd[b][k]] + " | " \
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


def get_board(boards: np.ndarray, index: int, view: int) -> np.ndarray:
    """
    Function to get a sub-board from nine board.
    :param boards: nine board, shape(10, 10).
    :param index: index of row.
    :param view: from my view or opponent's view, 0 (me) and 1 (opponent).
    :return: sub-board with shape (3, 3), with -1, 0, 1. -1 is opponent's pieces, 0 is empty, 1 is my pieces.
    """
    if 1 <= index <= 9:
        board = deepcopy(boards[index, 1:])
        if view == 0:
            board[np.where(board == 1)] = -1
            board[np.where(board == 0)] = 1
        else:
            board[np.where(board == 0)] = -1
            board[np.where(board == 1)] = 1
        board[np.where(board == 2)] = 0
        return board.reshape(3, 3)
    else:
        raise IndexError(f"Index {index} out of range.")


def check_winner_single_board(board: np.ndarray) -> int | None:
    """
    Function to check winner in one sub-board.
    :param board: sub-board, shape(3, 3).
    :return: 0, 1 or None. 0 means I win, 1 means opponent wins, None means draw or game not end yet.
    """
    for row in board:
        if np.all(row == 1):
            return 0
        elif np.all(row == -1):
            return 1
    for col in board.T:
        if np.all(col == 1):
            return 0
        elif np.all(col == -1):
            return 1
    diag1 = np.diag(board)
    diag2 = np.diag(np.fliplr(board))
    if np.all(diag1 == 1) or np.all(diag2 == 1):
        return 0
    elif np.all(diag1 == -1) or np.all(diag2 == -1):
        return 1
    return None


def check_draw(boards: np.ndarray, curr: int) -> bool:
    """
    Function to check whether the game is draw.
    :param boards: nine board, shape(10, 10)
    :param curr: index of current board.
    :return: True or False.
    """
    return len(np.where(boards[curr, 1:] != 2)[0]) == 9


def status(boards: np.ndarray, curr: int) -> (bool, int | None):
    """
    Function to check the status of game.
    :param boards: nine board, shape(10, 10)
    :param curr: index of current board.
    :return: [0]: True means draw or someone wins, False means game not end yet.
             [1]: see function "check_winner_single_board".
    """
    for i in range(1, 10):
        board = get_board(boards, i, 0)
        winner = check_winner_single_board(board)
        if winner is not None:
            return True, winner
    return check_draw(boards, curr), None


def board_to_str(board: np.ndarray) -> str:
    """
    Function to convert board to a string.
    :param board: board with shape(3, 3) or shape(10,).
    :return: A string represents the board.
    """
    if len(board.shape) == 2:
        board = board.reshape(-1, )
    elif board.shape[0] == 10:
        board = board[1:]
    return ''.join([['0', '1', '2'][int(i)] for i in board])


def generate_table(board: np.ndarray, score: np.ndarray) -> dict:
    """
    Function to create a Python dictionary to store boards and corresponding heuristic value,
    precomputation with the purpose of accelerate tree search.
    :param board: see function board_to_str.
    :param score: shape(n, 1).
    :return: dict Object.
    """
    return {board_to_str(board[i]): float(score[i, 0]) for i in range(len(board))}


def valid_movement(boards: np.ndarray, curr: int) -> list:
    """
    Function to check the valid action in given board.
    :param boards: nine board, shape(10, 10)
    :param curr: index of current board.
    :return: list of valid action.
    """
    return list(np.where(boards[curr] == 2)[0][1:])


if __name__ == '__main__':
    pass
