#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import torch
import numpy as np
import matplotlib.pyplot as plt
from env import Env, Game
from MCTS import MCTSPlayer
from MCTS_AZ import AlphaZeroPlayer
from Network import PolicyValueNet, PolicyValueNetQ
from ReplayBuffer import ReplayBuffer, ReplayBufferQ
from utils import inspect, inspectQ, set_learning_rate, instant_augment, instant_augmentQ
from elo import Elo

torch.set_float32_matmul_precision('high')


class TrainPipeline:
    def __init__(self, name='AlphaZero'):
        self.env = Env()
        self.game = Game(self.env)
        self.name = name
        self.params = './params'
        self.record = f'./result/{self.name}_eval.txt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(self.record, mode='w'):
            pass

    def init(self):
        params = f'{self.params}/{self.name}_current.pt'
        self.buffer = ReplayBuffer(3, self.buffer_size, 7)
        self.policy_value_net = PolicyValueNet(self.lr, params, self.device)
        self.az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)
        self.buffer.to(self.policy_value_net.device)
        self.elo = Elo(self.init_elo, 1500)
        self.R_A = [self.init_elo,]
        self.R_B = [1500,]
        self.best_elo = self.init_elo
        input('Confirm to continue.')

    def collect_selfplay_data(self, n_games=1):
        self.az_player.train()
        with torch.no_grad():
            for _ in range(n_games):
                _, play_data = self.game.start_self_play(
                    self.az_player, temp=self.temp, first_n_steps=self.first_n_steps, discount=self.discount, dirichlet_alpha=self.dirichlet_alpha)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                for data in play_data:
                    self.buffer.store(*data)

    @staticmethod
    def explained_var(pred, target):
        target = target.cpu().numpy().flatten()
        return 1 - np.var(target - pred.flatten()) / np.var(target)

    def policy_update(self):
        loss, entropy = [], []
        batch = self.buffer.sample(self.batch_size)
        batch = instant_augment(batch)
        old_probs, old_v = self.policy_value_net.policy_value(batch[0])
        for _ in range(self.epochs):
            set_learning_rate(self.policy_value_net.opt,
                              self.lr * self.lr_multiplier)
            res = self.policy_value_net.train_step(batch)
            new_probs, new_v = self.policy_value_net.policy_value(batch[0])
            loss.append(res[0])
            entropy.append(res[1])
            kl = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = self.explained_var(old_v, batch[-1])
        explained_var_new = self.explained_var(new_v, batch[-1])
        print(f'kl: {kl: .5f}\n'
              f'lr_multiplier: {self.lr_multiplier: .3f}\n'
              f'explained_var_old: {explained_var_old: .3f}\n'
              f'explained_var_new: {explained_var_new: .3f}')
        return np.mean(loss), np.mean(entropy)
    
    def evaluation(self, description, player1, player2, win_counter, win_key, lose_key, draw_key, n_games):
        print(description)
        for _ in range(n_games // 2):
            winner = self.game.start_play(player1=player1, player2=player2, show=0)
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
        inspect(self.policy_value_net.net)
        current_az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn,
                                            self.c_puct,
                                            self.n_playout)
        current_az_player.eval()
        mcts_player = MCTSPlayer(5, self.pure_mcts_n_playout)
        win_counter = {'Xwin': 0, 'Xdraw': 0, 'Xlose': 0,
                       'Owin': 0, 'Odraw': 0, 'Olose': 0}
        win_counter = self.evaluation('Evaluating policy X...', current_az_player, mcts_player, win_counter, 'Xwin', 'Xlose', 'Xdraw', n_games)
        win_counter = self.evaluation('Evaluating policy O...', mcts_player, current_az_player, win_counter, 'Owin', 'Olose', 'Odraw', n_games)
        self.elo.update(1 if 'Xwin' else 0.5 if 'Xdraw' else 0)
        return self.elo.update(1 if 'Owin' else 0.5 if 'Odraw' else 0)

    def run(self):
        current = f'{self.params}/{self.name}_current.pt'
        best = f'{self.params}/{self.name}_best.pt'
        img = f'./result/{self.name}_elo.jpg'
        for i in range(self.game_batch_num):
            self.collect_selfplay_data(self.play_batch_size)
            loss, entropy = float('inf'), float('inf')
            if len(self.buffer) > self.batch_size * 10:
                loss, entropy = self.policy_update()
            print(f'batch i: {i + 1}, episode_len: {self.episode_len}, '
                  f'loss: {loss: .8f}, entropy: {entropy: .8f}')
            if (i) % self.check_freq != 0:
                continue
            print(f'current self-play batch: {i + 1}')
            while True:
                r_a, r_b = self.policy_evaluate()
                self.R_A.append(r_a)
                self.R_B.append(r_b)
                plt.clf()
                plt.plot(self.R_A, label='Elo score (AlphaZero)')
                plt.plot(self.R_B, label=f'Elo score (MCTS: {self.pure_mcts_n_playout})')
                plt.grid(linestyle='--', alpha=0.3)
                plt.legend()
                plt.title(f'AlphaZero: {r_a}\nMCTS: {r_b}')
                plt.tight_layout()
                plt.savefig(img)
                self.policy_value_net.save(current)
                if r_a > self.best_elo:
                    print('New best policy!!')
                    self.best_elo = r_a
                    self.policy_value_net.save(best)


class TrainPipeline_Q_NewEval(TrainPipeline):
    def init(self):
        params = f'{self.params}/{self.name}_current.pt'
        self.buffer = ReplayBufferQ(3, self.buffer_size, 7)
        self.policy_value_net = PolicyValueNetQ(self.lr, params, self.device)
        self.az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)
        self.buffer.to(self.policy_value_net.device)
        input('Confirm to continue.')
    
    def collect_selfplay_data(self, n_games=1):
        self.az_player.train()
        with torch.no_grad():
            for _ in range(n_games):
                _, play_data = self.game.start_self_play_Q(
                    self.az_player, temp=self.temp, first_n_steps=self.first_n_steps, discount=self.discount, dirichlet_alpha=self.dirichlet_alpha)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                for data in play_data:
                    self.buffer.store(*data)
    
    def policy_evaluate(self, n_games=12):
        self.policy_value_net.eval()
        inspectQ(self.policy_value_net.net)
        current_az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn,
                                            self.c_puct,
                                            self.n_playout)
        current_az_player.eval()
        mcts_player = MCTSPlayer(5, self.pure_mcts_n_playout)
        win_counter = {'Xwin': 0, 'Xdraw': 0, 'Xlose': 0,
                       'Owin': 0, 'Odraw': 0, 'Olose': 0}
        win_counter = self.evaluation('Evaluating policy X...', current_az_player, mcts_player, win_counter, 'Xwin', 'Xlose', 'Xdraw', n_games)
        win_counter = self.evaluation('Evaluating policy O...', mcts_player, current_az_player, win_counter, 'Owin', 'Olose', 'Odraw', n_games)
        self.elo.update(1 if 'Xwin' else 0.5 if 'Xdraw' else 0)
        return self.elo.update(1 if 'Owin' else 0.5 if 'Odraw' else 0)
    
    def policy_update(self):
        loss, entropy = [], []
        batch = self.buffer.sample(self.batch_size)
        batch = instant_augmentQ(batch)
        old_probs, old_v = self.policy_value_net.policy_value(batch[0])
        for _ in range(self.epochs):
            set_learning_rate(self.policy_value_net.opt,
                              self.lr * self.lr_multiplier)
            res = self.policy_value_net.train_step(batch)
            new_probs, new_v = self.policy_value_net.policy_value(batch[0])
            loss.append(res[0])
            entropy.append(res[1])
            kl = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        explained_var_old = self.explained_var(old_v, batch[-1])
        explained_var_new = self.explained_var(new_v, batch[-1])
        print(f'kl: {kl: .5f}\n'
              f'lr_multiplier: {self.lr_multiplier: .3f}\n'
              f'explained_var_old: {explained_var_old: .3f}\n'
              f'explained_var_new: {explained_var_new: .3f}')
        return np.mean(loss), np.mean(entropy)
