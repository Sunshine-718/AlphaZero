#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024  13:03
import numpy as np
from ..Environment import Environment
from .utils import check_winner, valid_move, place, check_full, board_to_state, valid_mask


class Env(Environment):
    def __init__(self, board=None):
        super().__init__()
        self.board = np.zeros((6, 7), dtype=np.float32) if board is None else board
        self.turn = 1 if len(np.where(self.board != 0))[0] % 2 == 0 else -1
    
    def copy(self):
        new_env = Env()
        new_env.board = np.copy(self.board)
        new_env.turn = self.turn
        return new_env

    def reset(self):
        self.__init__()
        return self.board

    def done(self):
        return self.check_full() or self.winPlayer() != 0

    def valid_move(self):
        return valid_move(self.board)
    
    def valid_mask(self):
        return valid_mask(self.board)

    def switch_turn(self):
        self.turn = [0, -1, 1][self.turn]
        return self.turn

    def place(self, action):
        return place(self.board, action, self.turn)

    def check_full(self):
        return check_full(self.board)

    def winPlayer(self):
        return check_winner(self.board)

    def current_state(self):
        return board_to_state(self.board, self.turn)

    def step(self, action):
        if self.place(action):
            self.switch_turn()

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

    def flip(self, inplace: bool = False):
        target = self if inplace else self.copy()
        target.board = target.board[:, ::-1]
        return target

    def flip_action(self, col: int) -> int:
        return self.board.shape[1] - 1 - col

    def random_flip(self, p: float = 0.5):
        env_copy = self.copy()
        flipped = False
        if np.random.rand() < p:
            env_copy.board = env_copy.board[:, ::-1]
            flipped = True
        return env_copy, flipped
