#!/usr/bin/python3
import numpy as np
from .utils import generate_table, board_to_str, get_board, valid_movement, check_winner_single_board
import itertools


def evalBoard(board: np.ndarray, c: int | float = 2) -> int | float:
    """
    Function to evaluate the heuristic value of the sub-board, from Player agent.py's view, not lookt.
    :param board: shape(3, 3)
    :param c: a constant of bonus if two pieces of same player connected
    :return: heuristic value of the sub-board
    """
    win = check_winner_single_board(board)  # See the document of function from utils.py
    if win == 0:
        return 1000  # if player X wins score = 1000
    elif win == 1:
        return -1000  # if player O wins score = -1000
    score = 0
    for k in range(4):
        # rotate 90 degree for 4 times (0, -90, -180, -270), because the board is symmetry
        rotated = np.rot90(board, k, (0, 1))
        if rotated[0, 0] != 0:
            diag = np.diag(rotated)
            # each of the value can be -1 (opponent), 0 (empty), 1 (myself).
            if -rotated[0, 0] not in rotated[0, :]:  # Only X or only O exists in the first row
                if len(np.where(rotated[0, :] == rotated[0, 0])[0]) == 1:  # number of X or O is 1
                    # [*] if rotated[0, 0] = 1, score += 1 * 0.5, else if rotated[0, 0] = -1, score -= -1 * 0.5
                    score += rotated[0, 0] * 0.5
                else:  # number of X or O > 1
                    # All lines below like "score += rotated[i, j] * number" has the same rule as comment [*] above
                    score += rotated[0, 0]
            if -rotated[0, 0] not in rotated[:, 0]:  # Only X or only O exists in the first column
                if len(np.where(rotated[:, 0] == rotated[0, 0])[0]) == 1:  # number of X or O is 1
                    score += rotated[0, 0] * 0.5
                else:  # number of X or O > 1
                    score += rotated[0, 0]
            if -rotated[0, 0] not in diag:  # Only X or only O exists in the main diagnoal,
                # do not need to take secondary diagnoal into account because the board will rotate.
                score += rotated[0, 0] * 0.5
            if rotated[0, 0] == rotated[0, 1] and rotated[0, 0] != -rotated[0, 2]:
                # Only X or only O exists in the first row and two pieces are connected
                score += rotated[0, 0] * c
            if rotated[0, 0] == rotated[1, 1] and rotated[0, 0] != -rotated[2, 2]:
                # Only X or only O exists in the diagnoal and two pieces are connected
                score += rotated[0, 0] * c
            if rotated[0, 0] == rotated[1, 0] and rotated[0, 0] != -rotated[2, 0]:
                # Only X or only O exists in the first column and two pieces are connected
                score += rotated[0, 0] * c
        if rotated[0, 0] == rotated[0, 2] and rotated[0, 0] != -rotated[0, 1]:
            if rotated[1, 1] == 0:
                if rotated[0, 0] != -rotated[2, 0] and rotated[0, 0] != -rotated[2, 2]:
                    # detect the shape like
                    # X _ X    O _ O
                    # _ _ _ or _ _ _
                    # X _ X    X _ O
                    score += rotated[0, 0] * c
                elif rotated[0, 0] != -rotated[2, 0] or rotated[0, 0] != -rotated[2, 2]:
                    # detect the shape like
                    # X _ X    O _ O
                    # _ _ _ or _ _ _
                    # _ _ X    _ _ O
                    score += rotated[0, 0] * c
                else:
                    # detect the shape like
                    # X _ X    O _ O
                    # _ _ _ or _ _ _
                    # _ _ _    _ _ _
                    score += rotated[0, 0] * (c - 2)
        if rotated[0, 1] != 0:
            if -rotated[0, 1] not in rotated[:, 1]:  # second coloum
                if len(np.where(rotated[:, 1] == rotated[0, 1])[0]) == 1:
                    score += rotated[0, 1] * 0.5
                else:
                    score += rotated[0, 1]
            if -rotated[0, 1] not in rotated[0, :]:  # first row
                if len(np.where(rotated[0, :] == rotated[0, 1])[0]) == 1:
                    score += rotated[0, 1] * 0.5
                else:
                    score += rotated[0, 1]
            if rotated[1, 1] == rotated[0, 1] and rotated[0, 1] != -rotated[2, 1]:  # second coloum has connected pieces
                score += rotated[1, 1] * c
    diag1 = np.diag(board)
    diag2 = np.diag(np.fliplr(board))
    if board[1, 1] != 0:
        if -board[1, 1] not in board[:, 1]:  # second coloum
            score += board[1, 1]
        if -board[1, 1] not in board[1, :]:  # second row
            score += board[1, 1]
        if -board[1, 1] not in diag1:  # main diagnoal
            score += board[1, 1]
        if -board[1, 1] not in diag2:  # secondary diagnoal
            score += board[1, 1]
    else:
        # two disonnnected pieces
        if board[1, 0] == board[1, 2]:
            score += board[1, 0] * (c - 1)
        if board[0, 1] == board[2, 1]:
            score += board[0, 1] * (c - 1)
        if board[0, 0] == board[2, 2]:
            score += board[0, 0] * (c - 0.5)
        if board[0, 2] == board[2, 0]:
            score += board[0, 2] * (c - 0.5)
    return score


def evaluate_board(board: np.ndarray, c: int | float = 2) -> int | float:
    """
    See function evalBoard
    """
    return evalBoard(board, c)


Boards = np.array(list(itertools.product([-1, 0, 1], repeat=9)), dtype=np.int8)
heuristic_value = np.array([evaluate_board(i.reshape(3, 3)) for i in Boards]).reshape(-1, 1)
table = generate_table(Boards, heuristic_value)  # See the document of function "generate_table" from utils.py


class Heuristic:
    def __init__(self, table):
        self.table = table

    def action_heuristic(self, boards, curr: int, action: int, player: int) -> int | float:
        """
        Function to estimate action value given a state.
        :param curr: index of current board.
        :param action: action.
        :param player: 0 or 1, 0 means 'X', 1 means 'O'
        :return: value of given action.
        """
        before = self.estimate(boards, curr, player)
        boards[curr][action] = player
        after = self.estimate(boards, curr, player)
        boards[curr][action] = 2
        return after - before

    def sorted_action(self, boards, curr: int, branch: int = 9, player=None) -> tuple:
        """
        Function to sort action based on heuristic estimation, to accelerate tree search.
        :param curr: index of current board.
        :param branch: maximum node to expand at each level of tree, default 9 (maximum number of action in one board).
        :return: sorted sequence of action.
        """
        actions = list(
            reversed(
                sorted(valid_movement(boards, curr), key=lambda x: self.action_heuristic(boards, curr, x, player))))
        return tuple(actions[:min(branch, len(actions))])

    def estimate(self, boards, curr: int, player: int) -> float:
        """
        Function to get a heuristic estimate of nine boards.
        :param curr: index of current board.
        :param player: 0 or 1 (0 is X and 1 is O).
        :return: heuristic estimate of nine boards.
        """
        return np.mean([self.table[board_to_str(get_board(boards, i, player))] for i in range(1, 10)])


if __name__ == '__main__':
    board = np.array([[1, 1, 0],
                      [0, -1, 0],
                      [0, 0, 0]])
    print(evalBoard(board))
