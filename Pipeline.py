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
from tqdm.auto import tqdm
import multiprocessing as mp


def selfplay_worker(env_name, model_path, player_args, temp, first_n_steps):
    module = load(env_name)
    env = module.Env()
    game = Game(env)
    net = module.CNN(lr=0.0)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    policy_value_net = PolicyValueNet(net, model_path)
    az_player = AlphaZeroPlayer(policy_value_net, **player_args)
    _, play_data = game.start_self_play(
        az_player, temp=temp, first_n_steps=first_n_steps)
    return list(play_data)


def _preheat_task():
    return None


class TrainPipeline:
    def __init__(self, env_name='Connect4', model='CNN', name='AZ', use_multiprocessing=True, num_workers=None):
        """
        use_multiprocessing: bool, whether to use multiprocessing for self-play data collection.
        num_workers: int or None, number of worker processes to use. If None, defaults to play_batch_size.
        """
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
            self.net = self.module.CNN(lr=self.lr)
        elif model == 'ViT':
            self.net = self.module.ViT(lr=self.lr)
        else:
            raise ValueError(f'Unknown model type: {model}')
        params = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        self.policy_value_net = PolicyValueNet(self.net, params)
        self.az_player = AlphaZeroPlayer(self.policy_value_net, c_puct=self.c_puct,
                                         n_playout=self.n_playout, alpha=self.dirichlet_alpha, is_selfplay=1)
        self.update_best_player()
        self.current = f'{self.params}/{self.name}_{self.net.name()}_current.pt'
        self.best = f'{self.params}/{self.name}_{self.net.name()}_best.pt'
        self.elo = Elo(self.init_elo, 1500)
        if not os.path.exists('params'):
            os.makedirs('params')
        self.num_workers = num_workers if num_workers is not None else self.play_batch_size
        self.use_multiprocessing = use_multiprocessing and self.num_workers > 1
        if self.use_multiprocessing:
            # 若 psutil 可用，可改为物理核数
            self.pool = mp.Pool(processes=self.num_workers)
            # 预热进程池，避免第一次任务特别慢
            self.pool.apply_async(_preheat_task).get(timeout=10)
        else:
            self.pool = None

    def data_collector(self, n_games=1):
        self.policy_value_net.eval()
        self.az_player.train()
        self.az_player.to('cpu')
        episode_lens = []
        model_path = self.current
        player_args = dict(
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            alpha=self.dirichlet_alpha,
            is_selfplay=1
        )

        if self.pool is None:
            # 单进程直接运行
            pbar = tqdm(total=n_games, desc="Self-play")
            play_data_list = []
            for _ in range(n_games):
                play_data = selfplay_worker(self.env_name, model_path, player_args, self.temp, self.first_n_steps)
                episode_lens.append(len(play_data))
                play_data_list.append(play_data)
                pbar.update()
            pbar.close()
        else:
            # 多进程异步运行
            pbar = tqdm(total=n_games, desc="Self-play")
            results = []
            play_data_list = []

            def _cb(play_data):
                episode_lens.append(len(play_data))
                play_data_list.append(play_data)
                pbar.update()

            for _ in range(n_games):
                async_res = self.pool.apply_async(
                    selfplay_worker,
                    args=(self.env_name, model_path, player_args, self.temp, self.first_n_steps),
                    callback=_cb
                )
                results.append(async_res)

            for res in results:
                res.wait()

            pbar.close()

        for play_data in play_data_list:
            for data in play_data:
                self.buffer.store(*data, self.global_step)
        if episode_lens:
            self.episode_len = int(sum(episode_lens) / len(episode_lens))
        else:
            self.episode_len = 0

    def __del__(self):
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                self.pool.close()
                self.pool.join()
        except Exception:
            pass

    def policy_update(self):
        self.policy_value_net.to(self.device)
        dataloader = self.buffer.dataloader(self.batch_size)

        p_l, v_l, ent, g_n, f1 = self.policy_value_net.train_step(
            dataloader, self.module.instant_augment, self.global_step)

        print(f'F1 score (new): {f1: .3f}')
        self.policy_value_net.to('cpu')
        return p_l, v_l, ent, g_n, f1

    def run(self):
        self.show_hyperparams()
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
            self.policy_value_net.save(self.current)

            print(f'batch i: {self.global_step}, episode_len: {self.episode_len}, '
                  f'loss: {p_loss + v_loss: .8f}, entropy: {entropy: .8f}')

            writer.add_scalar('Metric/Gradient Norm',
                              grad_norm, self.global_step)
            writer.add_scalar('Metric/F1 score', f1, self.global_step)
            writer.add_scalars(
                'Metric/Loss', {'Action Loss': p_loss, 'Value loss': v_loss}, self.global_step)
            writer.add_scalar('Metric/Entropy', entropy, self.global_step)
            writer.add_scalar('Metric/Episode length',
                              self.episode_len, self.global_step)

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

            flag, win_rate = self.select_best_player(self.num_eval)
            writer.add_scalar('Metric/win rate', win_rate, self.global_step)
            if flag:
                print('New best policy!!')
                best_counter += 1
                writer.add_scalar('Metric/Best policy',
                                  best_counter, self.global_step)
                self.policy_value_net.save(self.best)

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
