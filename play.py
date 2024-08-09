#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
from env import Env, Game
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_AZ import AlphaZeroPlayer as MCTSPlayer
from Network import PolicyValueNet
import argparse

parser = argparse.ArgumentParser(description='Play connect four with AlphaZero!')
parser.add_argument('-x', action='store_true', help='Play as X')
parser.add_argument('-o', action='store_true', help='Play as O')
parser.add_argument('-n', type=int, default=1000, help='Number of simulations before AlphaZero make an action')
parser.add_argument('--self_play', action='store_true', help='AlphaZero play against itself')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Human:
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
        az_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=4,
                               n_playout=args.n, is_selfplay=0)
        az_player.eval()
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=370)
        human = Human()
        if args.x and args.o:
            game.start_play(human, human, show=1)
        elif args.x:
            game.start_play(human, az_player, show=1)
        elif args.o:
            game.start_play(az_player, human, show=1)
        elif args.self_play and not (args.x or args.o):
            game.start_play(az_player, az_player, show=1)
        else:
            raise ValueError('Invalid option\n'
                             f"<usage> python3 human_play.py -x [-o] [-n {args.n}]")
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
