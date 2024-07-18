#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024  13:03
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


class Env:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.turn = 1

    def reset(self):
        self.__init__()
        return self.board
    
    def done(self):
        return self.check_draw() or self.winPlayer() != 0

    def state(self):
        return board_to_state(self.board, self.turn)

    def valid_move(self):
        return valid_move(self.board)

    def valid_bin(self):
        return valid_bin(self.board)

    def switch_turn(self):
        self.turn = [0, -1, 1][self.turn]
        return self.turn

    def place(self, action):
        return place(self.board, action, self.turn)

    def check_draw(self):
        return check_draw(self.board)

    def winPlayer(self):
        return check_winner(self.board)

    def current_state(self):
        return board_to_state(self.board, self.turn)

    def step(self, action):
        next_step = step(self.board, action, self.turn)
        if next_step[-1]:
            self.switch_turn()
        return next_step

    def show(self):
        board = self.board.astype(int)
        temp = np.zeros_like(board, dtype=str)
        temp[np.where(board == 0)] = '_'
        temp[np.where(board == 1)] = 'X'
        temp[np.where(board == -1)] = 'O'
        print('=' * 20)
        for i in temp:
            print(' '.join(i))
        print(' '.join(map(str, range(7))))
        print('=' * 20)


class Game:
    def __init__(self, env):
        self.env = env

    def start_play(self, player1, player2, show=1):
        self.env.reset()
        players = [None, player1, player2]
        if show:
            self.env.show()
        while True:
            current_turn = self.env.turn
            player = players[current_turn]
            action = player.get_action(self.env)
            _, _, done, _ = self.env.step(action[0])
            if show:
                self.env.show()
            if done:
                winner = self.env.winPlayer()
                if show:
                    if winner != 0:
                        print('Game end. Winner is', [None, 'X', 'O'][int(winner)])
                    else:
                        print('Game end. Draw')
                return winner

    def start_self_play(self, player, temp=1e-3, first_n_steps=5, show=0, discount=0.99, dirichlet_alpha=0.3):
        self.env.reset()
        states, mcts_probs, current_players = [], [], []
        steps = 0
        while True:
            if steps < first_n_steps:
                action, probs = player.get_action(self.env, temp, dirichlet_alpha)
            else:
                action, probs = player.get_action(self.env, 1e-3, dirichlet_alpha)
            steps += 1
            states.append(self.env.current_state())
            mcts_probs.append(probs)
            current_players.append(self.env.turn)
            _, _, done, _ = self.env.step(action)
            if show:
                self.env.show()
            if done:
                winner = self.env.winPlayer()
                winner_z = np.zeros(len(current_players))
                if winner != 0:
                    winner_z[np.array(current_players) == winner] = 1
                    winner_z[np.array(current_players) != winner] = -1
                    for idx, i in enumerate(winner_z):
                        winner_z[idx] = i * pow(discount, len(winner_z) - idx - 1)
                if show:
                    if winner != 0:
                        print(f"Game end. Wineer is Player: {[None, 'X', 'O'][winner]}")
                    else:
                        print('Game end. Draw')
                return winner, zip(states, mcts_probs, winner_z)


if __name__ == '__main__':

    pass
