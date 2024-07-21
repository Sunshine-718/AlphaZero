#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import torch
import numpy as np
from env import Env, Game
from copy import deepcopy
from MCTS import MCTSPlayer
from MCTS_AZ import AlphaZeroPlayer
from Network import PolicyValueNet
from ReplayBuffer import ReplayBuffer
from tqdm.auto import tqdm
from utils import inspect, symmetric_state, set_learning_rate

torch.set_float32_matmul_precision('high')


class TrainPipeline:
    def __init__(self, name='AlphaZero'):
        self.env = Env()
        self.game = Game(self.env)
        self.name = name
        self.params = './params'
        self.record = f'./{self.name}_eval.txt'
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
        input('Confirm to continue.')

    def augment_data(self, play_data):
        extend_data = []
        for state, prob, winner in play_data:
            state_ = symmetric_state(state)
            prob_ = deepcopy(prob[::-1])
            extend_data.append((state_, prob_, winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        self.az_player.train()
        with torch.no_grad():
            for _ in range(n_games):
                _, play_data = self.game.start_self_play(
                    self.az_player, temp=self.temp, first_n_steps=self.first_n_steps, discount=self.discount, dirichlet_alpha=self.dirichlet_alpha)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                play_data.extend(self.augment_data(play_data))
                for data in play_data:
                    self.buffer.store(*data)

    @staticmethod
    def explained_var(pred, target):
        target = target.cpu().numpy().flatten()
        return 1 - np.var(target - pred.flatten()) / np.var(target)

    def policy_update(self):
        loss, entropy = [], []
        batch = self.buffer.sample(self.batch_size)
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
        iterator = tqdm(range(n_games // 2))
        iterator.set_description(description)
        for _ in iterator:
            winner = self.game.start_play(player1=player1, player2=player2, show=0)
            if winner != 0:
                if winner == 1:
                    win_counter[win_key] += 1
                else:
                    win_counter[lose_key] += 1
            else:
                win_counter[draw_key] += 1
        return win_counter


    def policy_evaluate(self, n_games=12):
        self.policy_value_net.eval()
        inspect(self.policy_value_net.net)
        current_az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn,
                                            self.c_puct,
                                            self.n_playout)
        current_az_player.eval()
        mcts_player = MCTSPlayer(5, self.pure_mcts_n_playout)
        win_counter = {'Xwin': 0, 'Xdraw': 0, 'Xlose': 0,
                       'Owin': 0, 'Odraw': 0, 'Olose': 0}
        win_counter = self.evaluate_policy('Evaluating policy X...', current_az_player, mcts_player, win_counter, 'Xwin', 'Xlose', 'Xdraw', n_games)
        win_counter = self.evaluate_policy('Evaluating policy O...', mcts_player, current_az_player, win_counter, 'Owin', 'Olose', 'Odraw', n_games)
        win_rate = (win_counter['Xwin'] + win_counter['Owin'] + 0.5 *
                     (win_counter['Xdraw'] + win_counter['Odraw'])) / n_games
        X_win_rate = (
            win_counter['Xwin'] + win_counter['Xdraw'] * 0.5) / (n_games // 2) * 100
        O_win_rate = (
            win_counter['Owin'] + win_counter['Odraw'] * 0.5) / (n_games // 2) * 100
        eval_res = (f"num_playouts: {self.pure_mcts_n_playout}\n"
                    f"\tX: win: {win_counter['Xwin']}, draw: {win_counter['Xdraw']}, lose: {win_counter['Xlose']}, win rate: {X_win_rate: .2f}%\n"
                    f"\tO: win: {win_counter['Owin']}, draw: {win_counter['Odraw']}, lose: {win_counter['Olose']}, win rate: {O_win_rate: .2f}%\n"
                    f"\ttotal:\n"
                    f"\twin: {win_counter['Xwin'] + win_counter['Owin']}, draw: {win_counter['Xdraw'] + win_counter['Odraw']}, lose: {win_counter['Xlose'] + win_counter['Olose']}, win rate: {win_rate * 100: .2f}%\n")
        print(eval_res, end='')
        with open(self.record, mode='a+') as f:
            f.write(eval_res)
        return win_rate

    def run(self):
        current = f'{self.params}/{self.name}_current.pt'
        best = f'{self.params}/{self.name}_best.pt'
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
                win_rate = self.policy_evaluate()
                self.policy_value_net.save(current)
                if win_rate > self.best_win_rate:
                    print('New best policy!!')
                    self.best_win_rate = win_rate
                    self.policy_value_net.save(best)
                    if (self.best_win_rate == 1.0 and self.pure_mcts_n_playout < 5000):
                        self.pure_mcts_n_playout += 10
                        self.best_win_rate = 0
                        continue
                    elif (win_rate == 0 and self.pure_mcts_n_playout > 10):
                        self.pure_mcts_n_playout -= 10
                        self.best_win_rate = 0
                if win_rate != 1.0:
                    break
