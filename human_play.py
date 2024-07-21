#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
from env import Env, Game
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_AZ import AlphaZeroPlayer as MCTSPlayer
from Network import PolicyValueNet


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        move = int(input('Your move: '))
        return move, None

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    params = './params/AlphaZero_current.pt'
    try:
        env = Env()
        game = Game(env)
        policy_value_net = PolicyValueNet(0, params, device)
        az_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5,
                               n_playout=100, is_selfplay=0)
        az_player.eval()
        mcts_player = MCTS_Pure(c_puct=5, n_playout=370)
        # human = Human()

        game.start_play(az_player, mcts_player, show=1)
        # game.start_play(mcts_player, az_player, show=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
