#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 13/Sep/2024  04:53
import numpy as np
from ..Environment import Environment
from .utils import print_board, winPlayer, valid_move, count_pieces, board_to_state


class Env(Environment):
    def __init__(self, init_board=None, init_action=None):
        super().__init__()
        self.boards = np.array([2 for _ in range(100)]).reshape(10, 10)
        self.turn = 1
        self.curr = np.random.randint(9) if init_board is None else init_board
        self.action = np.random.randint(9) if init_action is None else init_action
        _init = self.curr
        self.step(self.action)
        _valid = self.valid_move()
        if _init in _valid:
            _valid.remove(_init)
        self.step(np.random.choice(_valid))
        self.count = count_pieces(self.boards)

    def reset(self):
        self.__init__()
        return self.boards

    def done(self):
        return self.check_full() or self.winPlayer() != 0

    def valid_move(self):
        return valid_move(self.boards, self.curr)

    def switch_turn(self):
        self.turn = [0, -1, 1][self.turn]
        return self.turn

    def place(self, curr, action):
        if self.boards[curr + 1, action + 1] == 2:
            if self.turn == 1:
                self.boards[curr + 1, action + 1] = 0
            else:
                self.boards[curr + 1, action + 1] = 1
            self.curr = action
            return True
        return False

    def check_full(self):
        return not bool(len(np.where(self.boards[1:, 1:] == 2)[0]))

    def winPlayer(self):
        return winPlayer(self.boards)

    def current_state(self):
        return board_to_state(self.curr, self.boards, self.turn)

    def step(self, action):
        if self.place(self.curr, action):
            self.switch_turn()

    def show(self):
        print_board(self.boards)
