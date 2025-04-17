#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
import argparse
from environments import load
from game import Game
from policy_value_net import PolicyValueNet
from player import Human, AlphaZeroPlayer, NetworkPlayer


parser = argparse.ArgumentParser(
    description='Play connect four against AlphaZero!')
parser.add_argument('-x', action='store_true', help='Play as X')
parser.add_argument('-o', action='store_true', help='Play as O')
parser.add_argument('-n', type=int, default=500,
                    help='Number of simulations before AlphaZero make an action')
parser.add_argument('--self_play', action='store_true',
                    help='AlphaZero play against itself')
parser.add_argument('--model', type=str,
                    default='current', help='Model type')
parser.add_argument('--network', type=str, default='CNN', help='Network type')
parser.add_argument('--env', type=str, default='Connect4', help='env name')
parser.add_argument('--name', type=str, default='AZ2', help='Model name')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    module = load(args.env)
    config = module.config.training_config
    try:
        env = module.Env()
        game = Game(env)
        if args.network == 'CNN':
            net = module.CNN(0, device=device)
        elif args.network == 'ViT':
            net = module.ViT(0, device=device)
        policy_value_net = PolicyValueNet(
            net, config['discount'], f'./params/{args.name}_{args.env}_{args.network}_{args.model}.pt')
        if args.n == 0:
            az_player = NetworkPlayer(policy_value_net)
        else:
            az_player = AlphaZeroPlayer(policy_value_net, c_puct=config['c_puct'],
                                        n_playout=args.n, is_selfplay=0)
        az_player.eval()
        human = Human()
        if args.x and args.o:
            game.start_play(human, human, config['discount'], show=1)
        elif args.x:
            game.start_play(human, az_player, config['discount'], show=1)
        elif args.o:
            game.start_play(az_player, human, config['discount'], show=1)
        elif args.self_play and not (args.x or args.o):
            game.start_play(az_player, az_player, config['discount'], show=1)
        else:
            raise AttributeError('Invalid argument(s).\n'
                                 "Type 'python3 ./play.py -h' for help")
    except KeyboardInterrupt:
        print('\n\rquit')
