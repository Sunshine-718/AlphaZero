#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import torch
import numpy as np
from elo import Elo
from env import Env, Game
from Network import PolicyValueNet
from ReplayBuffer import ReplayBuffer
from config import network_config
from player import MCTSPlayer, AlphaZeroPlayer
from torch.utils.tensorboard import SummaryWriter
from utils import inspect, set_learning_rate, instant_augment


torch.set_float32_matmul_precision('high')


class TrainPipeline:
    def __init__(self, name='AlphaZero'):
        self.env = Env()
        self.game = Game(self.env)
        self.name = name
        self.params = './params'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init(self, config):
        for key, value in config.items():
            setattr(self, key, value)
        params = f'{self.params}/{self.name}_current.pt'
        self.buffer = ReplayBuffer(network_config['in_dim'], self.buffer_size, network_config['out_dim'])
        self.policy_value_net = PolicyValueNet(self.lr, self.discount, params, self.device, self.soft_update_rate)
        self.az_player = AlphaZeroPlayer(self.policy_value_net, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)
        self.buffer.to(self.policy_value_net.device)
        self.elo = Elo(self.init_elo, 1500)
        self.best_elo = self.init_elo

    def collect_selfplay_data(self, n_games=1):
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

    def policy_update(self, warm_up=False):
        p_loss, v_loss, entropy, grad_norm = [], [], [], []
        kl, ex_old, ex_new = [], [], []
        if warm_up:
            set_learning_rate(self.policy_value_net.opt, self.warmup_lr)
        else:
            self.lr = max(self.min_lr, self.lr * self.lr_discount)
            set_learning_rate(self.policy_value_net.opt, self.lr)
        for _ in range(self.epochs):
            batch = self.buffer.sample(self.batch_size)
            batch = instant_augment(batch)
            old_probs, old_v = self.policy_value_net.policy_value(batch[0])
            p_l, v_l, ent, g_n = self.policy_value_net.train_step(batch)
            new_probs, new_v = self.policy_value_net.policy_value(batch[0])
            p_loss.append(p_l)
            v_loss.append(v_l)
            entropy.append(ent)
            grad_norm.append(g_n)
            kl_temp = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)), axis=1))
            kl.append(kl_temp)
            ex_old.append(self.explained_var(old_v, batch[-1]))
            ex_new.append(self.explained_var(new_v, batch[-1]))
            if np.mean(kl) > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        kl = np.mean(kl)
        explained_var_old = np.mean(ex_old)
        explained_var_new = np.mean(ex_new)
        print(f'kl: {kl: .5f}\n'
              f'explained_var_old: {explained_var_old: .3f}\n'
              f'explained_var_new: {explained_var_new: .3f}')
        return np.mean(p_loss), np.mean(v_loss), np.mean(entropy), np.mean(grad_norm), explained_var_old, explained_var_new

    def evaluation(self, description, player1, player2, win_counter, win_key, lose_key, draw_key, n_games):
        print(description)
        for _ in range(n_games // 2):
            winner = self.game.start_play(
                player1=player1, player2=player2, show=0)
            if winner != 0:
                if winner == (1 if 'X' in win_key else -1):
                    win_counter[win_key] += 1
                else:
                    win_counter[lose_key] += 1
            else:
                win_counter[draw_key] += 1
        return win_counter

    def policy_evaluate(self, n_games=2):
        self.policy_value_net.eval()
        current_az_player = AlphaZeroPlayer(self.policy_value_net,
                                            self.c_puct,
                                            self.n_playout)
        current_az_player.eval()
        mcts_player = MCTSPlayer(5, self.pure_mcts_n_playout)
        win_counter = {'Xwin': 0, 'Xdraw': 0, 'Xlose': 0,
                       'Owin': 0, 'Odraw': 0, 'Olose': 0}
        win_counter = self.evaluation('Evaluating policy X...', current_az_player,
                                      mcts_player, win_counter, 'Xwin', 'Xlose', 'Xdraw', n_games)
        win_counter = self.evaluation('Evaluating policy O...', mcts_player,
                                      current_az_player, win_counter, 'Owin', 'Olose', 'Odraw', n_games)
        self.elo.update(1 if win_counter['Xwin']
                        else 0.5 if win_counter['Xdraw'] else 0)
        return self.elo.update(1 if win_counter['Owin'] else 0.5 if win_counter['Odraw'] else 0)
    
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
        print(f'\tInitial elo score: {self.init_elo}')
        print('=' * 50)

    def run(self):
        self.show_hyperparams()
        current = f'{self.params}/{self.name}_current.pt'
        best = f'{self.params}/{self.name}_best.pt'
        writer = SummaryWriter(filename_suffix=self.name)
        fake_input = torch.randn(1, 3, 6, 7).to(
            self.policy_value_net.net.device)
        writer.add_graph(self.policy_value_net.net, fake_input)
        writer.add_scalars('Metric/Elo', {f'AlphaZero: {self.n_playout}': self.init_elo,
                                          f'MCTS: {self.pure_mcts_n_playout}': 1500}, 0)
        preparing = True
        warm_up = True
        i = 0
        while True:
            self.collect_selfplay_data(self.play_batch_size)
            p_loss, v_loss, entropy, grad_norm = float(
                'inf'), float('inf'), float('inf'), float('inf')
            if len(self.buffer) > self.batch_size * 10:
                i += 1
                if preparing:
                    print(' ' * 100, end='\r')
                    print('Preparation phase completed.')
                    print('Start training...')
                    preparing = False
                if self.buffer.is_full():
                    warm_up = False
                    writer.add_scalar('Metric/Learning rate', self.lr, i)
                else:
                    writer.add_scalar('Metric/Learning rate', self.warmup_lr, i)
                p_loss, v_loss, entropy, grad_norm, ex_var_old, ex_var_new = self.policy_update(warm_up)
            else:
                perc = round(len(self.buffer) /
                             (self.batch_size * 10) * 100, 1)
                print(f'Preparing for training: {perc}%', end='\r')
                continue
            print(f'batch i: {i}, episode_len: {self.episode_len}, '
                  f'loss: {p_loss + v_loss: .8f}, entropy: {entropy: .8f}')
            writer.add_scalar('Metric/Gradient Norm', grad_norm, i)
            writer.add_scalars('Metric/Explained variance', {'Old': ex_var_old, 'New': ex_var_new}, i)
            if (i) % self.check_freq != 0:
                continue
            print(f'current self-play batch: {i + 1}')
            r_a, r_b = self.policy_evaluate()
            p0, v0, p1, v1 = inspect(self.policy_value_net.net)
            writer.add_scalars('Metric/Elo', {f'AlphaZero: {self.n_playout}': r_a,
                                              f'MCTS: {self.pure_mcts_n_playout}': r_b}, i)
            writer.add_scalars(
                'Metric/Loss', {'Action Loss': p_loss, 'Value loss': v_loss}, i)
            writer.add_scalar('Metric/Entropy', entropy, i)
            writer.add_scalar('Metric/Episode length', self.episode_len, i)
            writer.add_scalars('Metric/Initial Value', {'X': v0, 'O': v1}, i)
            writer.add_scalars('Action probability/X',
                               {str(idx): i for idx, i in enumerate(p0)}, i)
            writer.add_scalars('Action probability/O',
                               {str(idx): i for idx, i in enumerate(p1)}, i)
            self.policy_value_net.save(current)
            if r_a > self.best_elo:
                print('New best policy!!')
                self.best_elo = r_a
                writer.add_scalar('Metric/Highest elo', self.best_elo, i)
                self.policy_value_net.save(best)
