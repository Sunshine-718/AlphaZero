#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 25/Jun/2024  13:03
import numpy as np
from environments.Environment import Environment
from environments.Connect4.utils import check_winner, valid_move, place, check_draw, step, board_to_state


class Env(Environment):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((6, 7), dtype=np.float32)
        self.turn = 1

    def reset(self):
        self.__init__()
        return self.board

    def done(self):
        return self.check_draw() or self.winPlayer() != 0

    def valid_move(self):
        return valid_move(self.board)

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
