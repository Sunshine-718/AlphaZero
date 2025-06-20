#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024  13:03
import numpy as np
from ..Environment import Environment, BoardTransformerBase
from .utils import check_winner, valid_move, place, check_full, board_to_state, valid_mask


class BoardTransformer(BoardTransformerBase):
    def __init__(self, board, action_list):
        self._board = board
        self._action_list = action_list
    
    @property
    def board(self):
        return self._board

    def flip(self):
        self._action_list = self._action_list[::-1]
        self._board = self._board[:, ::-1]
    
    def get_action(self, action):
        return self._action_list[action]


class Env(Environment):
    def __init__(self, random_transform=True):
        super().__init__()
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.transformer = BoardTransformer(self.board, list(range(7)))
        if random_transform:
            self.random_transform()
        self.turn = 1
    
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
        return valid_move(self.transformer.board)
    
    def valid_mask(self):
        return valid_mask(self.transformer.board)

    def switch_turn(self):
        self.turn = [0, -1, 1][self.turn]
        return self.turn

    def place(self, action):
        return place(self.board, self.transformer.get_action(action), self.turn)

    def check_full(self):
        return check_full(self.board)

    def winPlayer(self):
        return check_winner(self.board)

    def current_state(self):
        return board_to_state(self.transformer.board, self.turn)

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
    
    def random_transform(self):
        if np.random.rand() < 0.5:
            self.transformer.flip()
        return self
    
    def __board_to_int(board):
        code = 0
        for i in range(6):
            for j in range(7):
                cell = board[i,j]
                val = 0 if cell == 0 else (1 if cell == 1 else 2)
                code = (code << 2) | val
        return code
    
    def __hash__(self):
        return self.__board_to_int(self.board)
