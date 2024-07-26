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


def evaluation(game, description, player1, player2, win_counter, win_key, lose_key, draw_key, n_games):

    iterator = tqdm(range(n_games // 2))
    iterator.set_description(description)
    for _ in iterator:
        winner = game.start_play(player1=player1, player2=player2, show=1)
        if winner != 0:
            if winner == (1 if 'X' in win_key else -1):
                win_counter[win_key] += 1
            else:
                win_counter[lose_key] += 1
        else:
            win_counter[draw_key] += 1
    return win_counter


def policy_evaluate(n_games=12):
    env = Env()
    game = Game(env)
    params1 = './params/AlphaZero_current.pt'
    net1 = PolicyValueNet(0, params1, device)
    net1.eval()
    az1 = MCTSPlayer(net1.policy_value_fn, c_puct=1.4,
                     n_playout=100, is_selfplay=0)
    az1.eval()
    mcts_player = MCTS_Pure(c_puct=5, n_playout=5000)
    win_counter = {'Xwin': 0, 'Xdraw': 0, 'Xlose': 0,
                   'Owin': 0, 'Odraw': 0, 'Olose': 0}
    # win_counter = evaluation(game, 'Evaluating policy X...', az1,
    #                          mcts_player, win_counter, 'Xwin', 'Xlose', 'Xdraw', n_games)
    win_counter = evaluation(game, 'Evaluating policy O...', mcts_player,
                             az1, win_counter, 'Owin', 'Olose', 'Odraw', n_games)
    win_rate = (win_counter['Xwin'] + win_counter['Owin'] + 0.5 *
                (win_counter['Xdraw'] + win_counter['Odraw'])) / n_games
    X_win_rate = (
        win_counter['Xwin'] + win_counter['Xdraw'] * 0.5) / (n_games // 2) * 100
    O_win_rate = (
        win_counter['Owin'] + win_counter['Odraw'] * 0.5) / (n_games // 2) * 100
    eval_res = (
        f"\tX: win: {win_counter['Xwin']}, draw: {win_counter['Xdraw']}, lose: {
            win_counter['Xlose']}, win rate: {X_win_rate: .2f}%\n"
        f"\tO: win: {win_counter['Owin']}, draw: {win_counter['Odraw']}, lose: {
            win_counter['Olose']}, win rate: {O_win_rate: .2f}%\n"
        f"\ttotal:\n"
        f"\twin: {win_counter['Xwin'] + win_counter['Owin']}, draw: {win_counter['Xdraw'] + win_counter['Odraw']}, lose: {win_counter['Xlose'] + win_counter['Olose']}, win rate: {win_rate * 100: .2f}%\n")
    print(eval_res, end='')
    return win_rate


def run():
    params1 = './params/AlphaZero_current.pt'
    params2 = './params/AlphaZero_current.pt'
    try:
        env = Env()
        game = Game(env)
        net1 = PolicyValueNet(0, params1, device)
        az1 = MCTSPlayer(net1.policy_value_fn, c_puct=1.4,
                         n_playout=1000, is_selfplay=0)
        az1.eval()
        net2 = PolicyValueNet(0, params2, device)
        az2 = MCTSPlayer(net2.policy_value_fn, c_puct=1.4,
                         n_playout=1000, is_selfplay=0)
        az2.eval()
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=2000)
        game.start_play(az1, az2, show=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
    # policy_evaluate(10)
