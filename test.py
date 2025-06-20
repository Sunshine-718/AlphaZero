#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Jul/2024  22:31
import torch
from environments import load
from game import Game
from policy_value_net import PolicyValueNet
from player import Human, AlphaZeroPlayer, NetworkPlayer, MCTSPlayer


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    module = load('Connect4')  # Change 'Connect4' to your desired environment
    config = module.config.training_config
    try:
        env = module.Env()
        game = Game(env)

        net = module.CNN(0, device=device)

        # policy_value_net = PolicyValueNet(
        #     net,
        #     config["discount"],
        #     f'./params/AZ2_Connect4_CNN_current.pt'
        # )

        # mcts = AlphaZeroPlayer(policy_value_net, c_puct=1.5,
        #                             n_playout=500, is_selfplay=0)
        # mcts.eval()
        
        mcts = MCTSPlayer(4, 1000, 1)
    
        game.start_play(mcts, mcts, 0.972, show=1, show_nn=0)
    except KeyboardInterrupt:
        print('\n\rquit')
        