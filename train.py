#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  21:00
import random
import torch
import numpy as np
from env import Env, Game
from copy import deepcopy
from MCTS import MCTSPlayer
from MCTS_AZ import AlphaZeroPlayer
from Network import PolicyValueNet
from ReplayBuffer import ReplayBuffer
from tqdm.auto import tqdm
from config import config

torch.set_float32_matmul_precision('high')


def symmetric_state(state):
    state = deepcopy(state)
    for idx, i in enumerate(state[0]):
        state[0, idx] = np.fliplr(i)
    return state


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class TrainPipeline:
    def __init__(self, params=None):
        self.env = Env()
        self.game = Game(self.env)
        self.params = params
    
    def init(self):
        self.buffer = ReplayBuffer(3, self.buffer_size, 7)
        self.policy_value_net = PolicyValueNet(
            self.lr, self.params, 'cuda' if torch.cuda.is_available() else 'cpu')
        self.az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
                                         n_playout=self.n_playout, is_selfplay=1)
        self.target_net = PolicyValueNet(0, self.params, self.policy_value_net.device)
        self.target_player = AlphaZeroPlayer(self.target_net.policy_value_fn, c_puct=self.c_puct,
                                             n_playout=self.n_playout, is_selfplay=1)
        self.buffer.to(self.policy_value_net.device)
    
    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, evaluation_param in zip(self.target_net.policy_value_net.parameters(), self.policy_value_net.policy_value_net.parameters()):
            target_param.data.copy_(tau * evaluation_param.data + (1 - tau) * target_param.data)

    def get_equi_data(self, play_data):
        extend_data = []
        for state, prob, winner in play_data:
            state_ = symmetric_state(state)
            prob_ = deepcopy(prob[::-1])
            extend_data.append((state_, prob_, winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        with torch.no_grad():
            for _ in range(n_games):
                _, play_data = self.game.start_self_play(
                    self.target_player, temp=self.temp, first_n_steps=self.first_n_steps, discount=self.discount)
                play_data = list(play_data)[:]
                self.episode_len = len(play_data)
                play_data.extend(self.get_equi_data(play_data))
                for data in play_data:
                    self.buffer.store(*data)

    def policy_update(self):
        loss, entropy = [], []
        batch = self.buffer.sample(self.batch_size)
        old_probs, old_v = self.policy_value_net.policy_value(batch[0])
        for _ in range(self.epochs):
            set_learning_rate(self.policy_value_net.opt, self.lr * self.lr_multiplier)
            res = self.policy_value_net.train_step(batch)
            new_probs, new_v = self.policy_value_net.policy_value(batch[0])
            loss.append(res[0])
            entropy.append(res[1])
            kl = np.mean(np.sum(
                old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:   # early stopping if D_KL diverges badly
                break
        self.soft_update(self.soft_update_rate)
        # adaptively adjust the learning rate
        # if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #     self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
        #     self.lr_multiplier *= 1.5
        explained_var_old = (
            1 - np.var(batch[-1].cpu().numpy().flatten() - old_v.flatten()) / np.var(batch[-1].cpu().numpy().flatten()))
        explained_var_new = (
            1 - np.var(batch[-1].cpu().numpy().flatten() - new_v.flatten()) / np.var(batch[-1].cpu().numpy().flatten()))
        print(f'kl: {kl: .5f}\nlr_multiplier: {self.lr_multiplier: .3f}\nexplained_var_old: {explained_var_old: .3f}\nexplain_var_new: {explained_var_new: .3f}')
        return np.mean(loss), np.mean(entropy)

    def policy_evaluate(self, n_games=10):
        current_az_player = AlphaZeroPlayer(self.policy_value_net.policy_value_fn,
                                            self.c_puct,
                                            self.n_playout)
        mcts_player = MCTSPlayer(1, self.pure_mcts_n_playout)
        win_counter = {'win': 0, 'draw': 0, 'lose': 0}
        iterator = tqdm(range(n_games // 2))
        iterator.set_description('Evaluating policy X...')
        for _ in iterator:
            winner = self.game.start_play(
                player1=current_az_player, player2=mcts_player, show=0)
            if winner != 0:
                if winner == 1:
                    win_counter['win'] += 1
                else:
                    win_counter['lose'] += 1
            else:
                win_counter['draw'] += 1
        iterator = tqdm(range(n_games // 2))
        iterator.set_description('Evaluating policy O...')
        for _ in iterator:
            winner = self.game.start_play(
                player1=mcts_player, player2=current_az_player, show=0)
            if winner != 0:
                if winner == -1:
                    win_counter['win'] += 1
                else:
                    win_counter['lose'] += 1
            else:
                win_counter['draw'] += 1
        win_ratio = (win_counter['win'] + 0.5 * win_counter['draw']) / n_games
        eval_res = f"num_playouts: {self.pure_mcts_n_playout}, win: {win_counter['win']}, draw: {win_counter['draw']}, lose: {win_counter['lose']}\n"
        print(eval_res)
        with open('./eval.txt', mode='a+') as f:
            f.write(eval_res)
        # self.buffer.reset()
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                loss, entropy = float('inf'), float('inf')
                if len(self.buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                print(f'batch i: {i + 1}, episode_len: {self.episode_len}, loss: {loss: .8f}, entropy: {entropy: .8f}')
                if (i) % self.check_freq == 0:
                    print(f'current self-play batch: {i + 1}')
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save('./params/current.pt')
                    if win_ratio > self.best_win_ratio:
                        print('New best policy!!')
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save('./params/best.pt')
                        if (self.best_win_ratio == 1.0 and self.pure_mcts_n_playout < 5000):
                            self.pure_mcts_n_playout += 50
                            self.best_win_ratio = 0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    pipeline = TrainPipeline('./params/current.pt')
    for key, value in config.items():
        setattr(pipeline, key, value)
    pipeline.init()
    pipeline.run()
    pass
