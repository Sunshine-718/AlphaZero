#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import torch
import numpy as np
from utils import Elo
from game import Game
from copy import deepcopy
from torchsummary import summary
from environments import load
from policy_value_net import PolicyValueNet
from ReplayBuffer import ReplayBuffer
from player import MCTSPlayer, AlphaZeroPlayer, NetworkPlayer
from torch.utils.tensorboard import SummaryWriter


torch.set_float32_matmul_precision('high')


class TrainPipeline:
    def __init__(self, env_name, name='AlphaZero'):
        collection = ('Connect4', 'NBTTT')
        if env_name not in collection:
            raise ValueError(f'Environment does not exist, available env: {collection}')
        self.env_name = env_name
        self.module = load(env_name)
        self.env = self.module.Env()
        self.game = Game(self.env)
        self.name = f'{name}_{env_name}'
        self.params = './params'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for key, value in self.module.training_config.items():
            setattr(self, key, value)
        params = f'{self.params}/{self.name}_current.pt'
        self.buffer = ReplayBuffer(self.module.network_config['in_dim'], 
                                   self.buffer_size, 
                                   self.module.network_config['out_dim'], 
                                   self.module.env_config['row'], 
                                   self.module.env_config['col'])
        self.net = self.module.Network(self.lr, 
                                       self.module.network_config['in_dim'],
                                       self.module.network_config['h_dim'],
                                       self.module.network_config['out_dim'],
                                       self.device)
        self.policy_value_net = PolicyValueNet(self.net, self.discount, params)
        self.az_player = AlphaZeroPlayer(self.policy_value_net, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)
        self.update_best_player()
        self.buffer.to(self.policy_value_net.device)
        self.elo = Elo(self.init_elo, 1500)       
    
    def update_best_player(self):
        self.best_net = deepcopy(self.policy_value_net)
        self.best_player = AlphaZeroPlayer(self.best_net, c_puct=self.c_puct,
                                         n_playout=self.n_playout)

    def collect_selfplay_data(self, n_games=1):
        self.policy_value_net.eval()
        self.az_player.train()
        with torch.no_grad():
            for _ in range(n_games):
                _, play_data = self.game.start_self_play(self.az_player, temp=self.temp, first_n_steps=self.first_n_steps, discount=self.discount, dirichlet_alpha=self.dirichlet_alpha)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                for data in play_data:
                    self.buffer.store(*data)

    @staticmethod
    def explained_var(pred, target):
        target = target.cpu().numpy().flatten()
        return 1 - np.var(target - pred.flatten()) / np.var(target)

    def policy_update(self):
        p_loss, v_loss, entropy, grad_norm = [], [], [], []
        for _ in range(self.epochs):
            batch = self.buffer.sample(self.batch_size)
            batch = self.module.instant_augment(batch)
            old_probs, old_v = self.policy_value_net.policy_value(batch[0])
            p_l, v_l, ent, g_n = self.policy_value_net.train_step(batch)
            new_probs, new_v = self.policy_value_net.policy_value(batch[0])
            p_loss.append(p_l)
            v_loss.append(v_l)
            entropy.append(ent)
            grad_norm.append(g_n)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)), axis=1))
            explained_var_old = self.explained_var(old_v, batch[2])
            explained_var_new = self.explained_var(new_v, batch[2])
        print(f'kl: {kl: .5f}\n'
              f'explained_var_old: {explained_var_old: .3f}\n'
              f'explained_var_new: {explained_var_new: .3f}')
        return np.mean(p_loss), np.mean(v_loss), np.mean(entropy), np.mean(grad_norm), explained_var_old, explained_var_new

    def update_elo(self):
        print('Updating elo score...')
        self.policy_value_net.eval()
        current_az_player = AlphaZeroPlayer(self.policy_value_net,
                                            self.c_puct,
                                            self.n_playout)
        current_az_player.eval()
        mcts_player = MCTSPlayer(5, self.pure_mcts_n_playout)
        winner = self.game.start_play(player1=current_az_player, player2=mcts_player, discount=self.discount, show=0)
        self.elo.update(1 if winner == 1 else 0.5 if winner == 0 else 0)
        winner = self.game.start_play(player1=mcts_player, player2=current_az_player, discount=self.discount, show=0)
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
            winner = self.game.start_play(player1=current_player, player2=best_player, discount=self.discount, show=0)
            if winner == 1:
                win_rate += 1 / n_games
            elif winner == 0:
                win_rate += 0.5 / n_games
        for _ in range(n_games // 2):
            winner = self.game.start_play(player1=best_player, player2=current_player, discount=self.discount, show=0)
            if winner == -1:
                win_rate += 1 / n_games
            elif winner == 0:
                win_rate += 0.5 / n_games
        if win_rate >= self.win_rate_threshold:
            self.update_best_player()
            flag = True
        print('Complete.')
        return flag, win_rate
    
    def __call__(self):
        self.run()
    
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
        print(f'\tDiscount: {self.discount}')
        print(f'\tTemperature: {self.temp}')
        print('=' * 50)

    def run(self):
        self.show_hyperparams()
        fake_input_shape = (self.module.network_config['in_dim'], 
                           self.module.env_config['row'], 
                           self.module.env_config['col'])
        summary(self.net, fake_input_shape, device=self.device)
        current = f'{self.params}/{self.name}_current.pt'
        best = f'{self.params}/{self.name}_best.pt'
        writer = SummaryWriter(filename_suffix=self.name)
        fake_input = torch.randn(1, *fake_input_shape).to(self.device)
        writer.add_graph(self.policy_value_net.net, fake_input)
        writer.add_scalars('Metric/Elo', {f'AlphaZero: {self.n_playout}': self.init_elo,
                                          f'MCTS: {self.pure_mcts_n_playout}': 1500}, 0)
        preparing = True
        i = 0
        best_counter = 0
        while True:
            self.collect_selfplay_data(self.play_batch_size)
            p_loss, v_loss, entropy, grad_norm = float('inf'), float('inf'), float('inf'), float('inf')
            if len(self.buffer) > self.batch_size * 10:
                i += 1
                if preparing:
                    print(' ' * 100, end='\r')
                    print('Preparation phase completed.')
                    print('Start training...')
                    preparing = False
                p_loss, v_loss, entropy, grad_norm, ex_var_old, ex_var_new = self.policy_update()
            else:
                perc = round(len(self.buffer) / (self.batch_size * 10) * 100, 1)
                print(f'Preparing for training: {perc}%', end='\r')
                continue
            print(f'batch i: {i}, episode_len: {self.episode_len}, '
                  f'loss: {p_loss + v_loss: .8f}, entropy: {entropy: .8f}')
            writer.add_scalar('Metric/Gradient Norm', grad_norm, i)
            writer.add_scalars('Metric/Explained variance', {'Old': ex_var_old, 'New': ex_var_new}, i)
            if (i) % 10 != 0:
                continue
            print(f'current self-play batch: {i + 1}')
            r_a, r_b = self.update_elo()
            writer.add_scalars('Metric/Elo', {f'AlphaZero: {self.n_playout}': r_a,
                                              f'MCTS: {self.pure_mcts_n_playout}': r_b}, i)
            writer.add_scalars(
                'Metric/Loss', {'Action Loss': p_loss, 'Value loss': v_loss}, i)
            writer.add_scalar('Metric/Entropy', entropy, i)
            writer.add_scalar('Metric/Episode length', self.episode_len, i)
            if self.env_name == 'Connect4':
                p0, v0, p1, v1 = self.module.inspect(self.policy_value_net.net)
                writer.add_scalars('Metric/Initial Value', {'X': v0, 'O': v1}, i)
                writer.add_scalars('Action probability/X',
                                {str(idx): i for idx, i in enumerate(p0)}, i)
                writer.add_scalars('Action probability/O',
                                {str(idx): i for idx, i in enumerate(p1)}, i)
                writer.add_scalars('Action probability/X_cummulative', 
                                {str(idx): i for idx, i in enumerate(np.cumsum(p0))}, i)
                writer.add_scalars('Action probability/O_cummulative',
                                {str(idx): i for idx, i in enumerate(np.cumsum(p1))}, i)
            self.policy_value_net.save(current)
            if (i) % 50 != 0:
                continue
            flag, win_rate = self.select_best_player(self.num_eval)
            writer.add_scalar('Metric/win rate', win_rate, i)
            if flag:
                print('New best policy!!')
                best_counter += 1
                writer.add_scalar('Metric/Best policy', best_counter, i)
                self.policy_value_net.save(best)
