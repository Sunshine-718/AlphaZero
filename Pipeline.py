#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import os
import torch
import numpy as np
from utils import Elo
from game import Game
from copy import deepcopy
from environments import load
from policy_value_net import PolicyValueNet
from ReplayBuffer import ReplayBuffer
from player import MCTSPlayer, AlphaZeroPlayer, NetworkPlayer
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange


class TrainPipeline:
    def __init__(self, env_name='Connect4', model='CNN', name='AZ'):
        collection = ('Connect4', )  # NBTTT implementation not yet finished.
        if env_name not in collection:
            raise ValueError(
                f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.name = f'{name}_{env_name}'
        self.params = './params'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.global_step = 0
        for key, value in self.module.training_config.items():
            setattr(self, key, value)
        self.buffer = ReplayBuffer(self.module.network_config['in_dim'],
                                   self.buffer_size,
                                   self.module.network_config['out_dim'],
                                   self.module.env_config['row'],
                                   self.module.env_config['col'],
                                   device=self.device)
        if model == 'CNN':
            self.net = self.module.CNN(lr=self.lr, device=self.device)
        elif model == 'ViT':
            self.net = self.module.ViT(lr=self.lr, device=self.device)
        else:
            raise ValueError(f'Unknown model type: {model}')
        params = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        self.policy_value_net = PolicyValueNet(self.net, params)
        self.az_player = AlphaZeroPlayer(self.policy_value_net, c_puct=self.c_puct,
                                         n_playout=self.n_playout, alpha=self.dirichlet_alpha, is_selfplay=1)
        self.update_best_player()
        self.elo = Elo(self.init_elo, 1500)
        if not os.path.exists('params'):
            os.makedirs('params')

    def data_collector(self, n_games=1):
        self.policy_value_net.eval()
        self.az_player.train()
        episode_len = []
        with torch.no_grad():
            for _ in trange(n_games):
                _, play_data = self.game.start_self_play(
                    self.az_player, temp=self.temp, first_n_steps=self.first_n_steps)
                play_data = list(play_data)[:]
                episode_len.append(len(play_data)) 
                for data in play_data:
                    self.buffer.store(*data, self.global_step)
        self.episode_len = int(np.mean(episode_len))

    def policy_update(self):
        dataloader = self.buffer.dataloader(self.batch_size)
        
        p_l, v_l, ent, g_n, f1 = self.policy_value_net.train_step(dataloader, self.module.instant_augment, self.global_step)
            
        print(f'F1 score (new): {f1: .3f}')
        return p_l, v_l, ent, g_n, f1

    def run(self):
        self.show_hyperparams()
        current = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        best = f'{self.params}/{self.name}_{self.net.name()}_best.pt'
        writer = SummaryWriter(filename_suffix=self.name)
        writer.add_scalars('Metric/Elo', {f'AlphaZero_{self.n_playout}': self.init_elo,
                                          f'MCTS_{self.pure_mcts_n_playout}': 1500}, 0)
        best_counter = 0
        while True:
            self.data_collector(self.play_batch_size)
            p_loss, v_loss, entropy, grad_norm = float('inf'), float('inf'), \
                float('inf'), float('inf')
            self.global_step += 1
            p_loss, v_loss, entropy, grad_norm, f1 = self.policy_update()
            
            print(f'batch i: {self.global_step}, episode_len: {self.episode_len}, '
                  f'loss: {p_loss + v_loss: .8f}, entropy: {entropy: .8f}')

            writer.add_scalar('Metric/Gradient Norm', grad_norm, self.global_step)
            writer.add_scalar('Metric/F1 score', f1, self.global_step)
            writer.add_scalars(
                'Metric/Loss', {'Action Loss': p_loss, 'Value loss': v_loss}, self.global_step)
            writer.add_scalar('Metric/Entropy', entropy, self.global_step)
            writer.add_scalar('Metric/Episode length', self.episode_len, self.global_step)

            if (self.global_step) % 10 != 0:
                continue

            print(f'current self-play batch: {self.global_step + 1}')
            r_a, r_b = self.update_elo()
            print(f'Elo score: AlphaZero: {r_a: .2f}, Benchmark: {r_b: .2f}')
            writer.add_scalars('Metric/Elo', {f'AlphaZero_{self.n_playout}': r_a,
                                              f'MCTS_{self.pure_mcts_n_playout}': r_b}, self.global_step)

            if self.env_name == 'Connect4':
                p0, v0, p1, v1 = self.module.inspect(self.policy_value_net.net)
                writer.add_scalars('Metric/Initial Value',
                                   {'X': v0, 'O': v1}, self.global_step)
                writer.add_scalars('Action probability/X',
                                   {str(idx): i for idx, i in enumerate(p0)}, self.global_step)
                writer.add_scalars('Action probability/O',
                                   {str(idx): i for idx, i in enumerate(p1)}, self.global_step)
                writer.add_scalars('Action probability/X_cummulative',
                                   {str(idx): i for idx, i in enumerate(np.cumsum(p0))}, self.global_step)
                writer.add_scalars('Action probability/O_cummulative',
                                   {str(idx): i for idx, i in enumerate(np.cumsum(p1))}, self.global_step)
            self.policy_value_net.save(current)

            flag, win_rate = self.select_best_player(self.num_eval)
            writer.add_scalar('Metric/win rate', win_rate, self.global_step)
            if flag:
                print('New best policy!!')
                best_counter += 1
                writer.add_scalar('Metric/Best policy', best_counter, self.global_step)
                self.policy_value_net.save(best)

    def update_elo(self):
        print('Updating elo score...')
        self.policy_value_net.eval()
        current_az_player = AlphaZeroPlayer(self.policy_value_net,
                                            self.c_puct,
                                            self.n_playout,
                                            self.dirichlet_alpha)
        current_az_player.eval()
        mcts_player = MCTSPlayer(1, self.pure_mcts_n_playout)
        winner = self.game.start_play(
            player1=current_az_player, player2=mcts_player, show=0)
        self.elo.update(1 if winner == 1 else 0.5 if winner == 0 else 0)
        winner = self.game.start_play(
            player1=mcts_player, player2=current_az_player, show=0)
        print('Complete.')
        return self.elo.update(1 if winner == -1 else 0.5 if winner == 0 else 0)

    def select_best_player(self, n_games=10):
        print('Evaluating best player...')
        self.policy_value_net.eval()
        self.best_net.eval()
        current_player = NetworkPlayer(self.policy_value_net, False)
        best_player = NetworkPlayer(self.best_net, False)
        current_player.eval()
        best_player.eval()
        win_rate = 0
        flag = False
        for _ in range(n_games // 2):
            winner = self.game.start_play(
                player1=current_player, player2=best_player, show=0)
            if winner == 1:
                win_rate += 1 / n_games
            elif winner == 0:
                win_rate += 0.5 / n_games
        for _ in range(n_games // 2):
            winner = self.game.start_play(
                player1=best_player, player2=current_player, show=0)
            if winner == -1:
                win_rate += 1 / n_games
            elif winner == 0:
                win_rate += 0.5 / n_games
        if win_rate >= self.win_rate_threshold:
            self.update_best_player()
            flag = True
        print('Complete.')
        return flag, win_rate

    def show_hyperparams(self):
        print('=' * 50)
        print('Hyperparameters:')
        print(f'\tC_puct: {self.c_puct}')
        print(f'\tSimulation (AlphaZero): {self.n_playout}')
        print(f'\tSimulation (Benchmark): {self.pure_mcts_n_playout}')
        print(f'\tDirichlet alpha: {self.dirichlet_alpha}')
        print(f'\tBuffer size: {self.buffer_size}')
        print(f'\tBatch size: {self.batch_size}')
        print(f'\tRandom steps: {self.first_n_steps}')
        print(f'\tTemperature: {self.temp}')
        print('=' * 50)

    def update_best_player(self):
        self.best_net = deepcopy(self.policy_value_net)
        self.best_player = AlphaZeroPlayer(self.best_net, c_puct=self.c_puct, 
                                           n_playout=self.n_playout, alpha=self.dirichlet_alpha)

    def __call__(self):
        self.run()
