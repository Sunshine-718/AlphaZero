#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 10/Aug/2024  23:47


class Environment:
    def __init__(self):
        self.board = None
        self.turn = None    # Recommended: 1 for X and -1 for O

    def reset(self):
        self.__init__()
        return self.board
    
    def done(self):
        """
        Function to check the game is done, including win, lose, draw
        """
        return self.check_draw() or self.winPlayer() != 0

    def valid_move(self):
        """
        Function to return valid action based on current board.
        """
        raise NotImplementedError

    def switch_turn(self):
        """
        Function to switch the player's turn, i.e., from X to O or from O to X.
        """
        raise NotImplementedError

    def place(self, *args, **kwargs):
        """
        Function to place the piece to the board
        """
        raise NotImplementedError

    def check_draw(self):
        """
        Function to check whether the game is draw.
        """
        raise NotImplementedError

    def winPlayer(self):
        """
        Function to check the winner.
        Recommended: 1: X wins, -1: O wins, 0: no one wins.
        """
        raise NotImplementedError

    def current_state(self):
        """
        Function to return the processed state as the input of the neural network.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """
        Function to place the piece and switch the turn.
        : Return: next_state
        """
        raise NotImplementedError

    def show(self):
        """
        Function to show the board.
        """
        raise NotImplementedError