# -*- coding: utf-8 -*-
# @Time: 2024/4/18 22:07
import numpy as np
from NegamaxAgent import NegamaxAgent
from utils import count_pieces, print_board, status
from Heuristic import Heuristic, table
import time

if __name__ == '__main__':
    boards = np.array([[2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    curr = 2
    action = curr
    player = 0
    depth = 3
    print_board(boards)
    heuristic = Heuristic(boards, table)
    agent0 = NegamaxAgent(boards, heuristic, 0)
    agent1 = NegamaxAgent(boards, heuristic, 1)
    while True:

        print(f'valid movement in board {curr}: {heuristic.sorted_action(curr, 9, player)}')
        print(f'Utility X: {heuristic.estimate(curr, 0): .2f}')
        print(f'Utility O: {heuristic.estimate(curr, 1): .2f}')
        # action = int(input(f"[Board {env.curr}], Input an action: "))
        step = count_pieces(boards)
        start = time.perf_counter()
        if player == 0:
            action = agent0.action(curr)
        else:
            action = agent1.action(curr)
        print(f'Process Time: {time.perf_counter() - start: .2f}s')
        boards[curr][action] = player
        print_board(boards)
        player = 1 - player
        curr = action
        if status(boards, curr)[0]:
            if status(boards, curr)[1] is None:
                print('Draw')
            else:
                print(f"Player {['X', 'O'][status(boards, curr)[1]]} win!")
                print(f'Utility X: {heuristic.estimate(curr, 0): .2f}')
                print(f'Utility O: {heuristic.estimate(curr, 1): .2f}')
            break
