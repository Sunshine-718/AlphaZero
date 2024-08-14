#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
import argparse
from config import config
from env import Env, Game
from Network import PolicyValueNet
from player import Human, MCTSPlayer, AlphaZeroPlayer, NetworkPlayer


parser = argparse.ArgumentParser(
    description='Play connect four against AlphaZero!')
parser.add_argument('-x', action='store_true', help='Play as X')
parser.add_argument('-o', action='store_true', help='Play as O')
parser.add_argument('-n', type=int, default=500,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--self_play', action='store_true',
                    help='AlphaZero play against itself')
parser.add_argument('--model', type=str,
                    default='./params/AlphaZero_current.pt', help='Model file path')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    try:
        env = Env()
        game = Game(env)
        policy_value_net = PolicyValueNet(0, config['discount'], args.model, device)
        if args.n == 0:
            az_player = NetworkPlayer(policy_value_net)
        else:
            az_player = AlphaZeroPlayer(policy_value_net.policy_value_fn, c_puct=config['c_puct'],
                                        n_playout=args.n, is_selfplay=0)
        az_player.eval()
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
            raise AttributeError('Invalid option\n'
                                 "Type 'python3 ./play.py -h' for help")
    except KeyboardInterrupt:
        print('\n\rquit')
