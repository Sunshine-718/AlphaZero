#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 24/Jul/2024  20:13
import torch
from env import Env, Game
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_AZ import AlphaZeroPlayer as MCTSPlayer
from Network import PolicyValueNet
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run():
    params1 = './params/AlphaZero_current.pt'
    params2 = './params/AlphaZero_current.pt'
    try:
        env = Env()
        game = Game(env)
        net1 = PolicyValueNet(0, params1, device)
        az1 = MCTSPlayer(net1.policy_value_fn, c_puct=1.5, n_playout=1000, is_selfplay=0)
        az1.eval()
        net2 = PolicyValueNet(0, params2, device)
        az2 = MCTSPlayer(net2.policy_value_fn, c_puct=1.5, n_playout=1000, is_selfplay=0)
        az2.eval()
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=2000)
        game.start_play(az1, az2, show=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
    # policy_evaluate(10)
