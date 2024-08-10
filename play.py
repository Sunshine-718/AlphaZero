#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
import argparse
from config import config
from player import Player
from env import Env, Game
from Network import PolicyValueNet
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_AZ import AlphaZeroPlayer as MCTSPlayer


parser = argparse.ArgumentParser(description='Play connect four against AlphaZero!')
parser.add_argument('-x', action='store_true', help='Play as X')
parser.add_argument('-o', action='store_true', help='Play as O')
parser.add_argument('-n', type=int, default=100, help='Number of simulations before AlphaZero make an action')
parser.add_argument('--self_play', action='store_true', help='AlphaZero play against itself')
parser.add_argument('--model', type=str, default='./params/AlphaZero_best.pt', help='Model file path')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Human(Player):
    def get_action(self, *args, **kwargs):
        move = int(input('Your move: '))
        return move, None


def run():
    try:
        env = Env()
        game = Game(env)
        policy_value_net = PolicyValueNet(0, args.model, device)
        az_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=config['c_puct'],
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
