import numpy as np
from numba import njit
import torch

@njit(fastmath=True)
def board_to_state(board, turn):
    temp = np.zeros((1, 3, 3, 3), dtype=np.float32)
    temp[:, 0] = board == 1
    temp[:, 1] = board == -1
    temp[:, 2] = turn
    return temp

@njit(fastmath=True)
def check_full(board):
    return not np.any(board == 0)

@njit(fastmath=True)
def check_winner(board):
    # 行、列
    for i in range(3):
        if board[i, 0] != 0 and np.all(board[i, :] == board[i, 0]):
            return board[i, 0]
        if board[0, i] != 0 and np.all(board[:, i] == board[0, i]):
            return board[0, i]
    # 对角线
    if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
        return board[0, 0]
    if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
        return board[0, 2]
    return 0

@njit
def valid_move(board):
    return [i for i in range(9) if board.flat[i] == 0]

@njit
def valid_mask(board):
    return [board.flat[i] == 0 for i in range(9)]

@njit
def place(board, action, turn):
    if board.flat[action] == 0:
        board.flat[action] = turn
        return True
    return False


def print_row(action, probX, probO, max_X, max_O):
    print('⭐️ ' if probX == max_X else '   ', end='')
    print(f'action: {action}, prob_X: {probX * 100: 02.2f}%', end='\t')
    print('⭐️ ' if probO == max_O else '   ', end='')
    print(f'action: {action}, prob_O: {probO * 100: 02.2f}%')


def inspect(net, board=None):
    if board is None:
        board = np.zeros((3, 3), dtype=np.float32)
    with torch.no_grad():
        state0 = torch.from_numpy(board_to_state(board, 1)).float().to(net.device)
        p0, value_quantiles0 = net(state0)
        probs0 = torch.exp(p0).detach().cpu().numpy().flatten()
        value0 = np.tanh(value_quantiles0.mean(dim=-1).item())  # 如果有量子头，否则用 .item()
        board[1, 1] = 1  # 随便放个子，让另一个玩家不是空盘
        state1 = torch.from_numpy(board_to_state(board, -1)).float().to(net.device)
        p1, value_quantiles1 = net(state1)
        probs1 = torch.exp(p1).detach().cpu().numpy().flatten()
        value1 = np.tanh(value_quantiles1.mean(dim=-1).item())
    for idx in range(9):
        print_row(idx, probs0[idx], probs1[idx], np.max(probs0), np.max(probs1))
    print(f'State-value X: {value0:.4f}\nState-value O: {value1:.4f}')
    return probs0, value0, probs1, value1


def instant_augment(batch):
    state, prob, value, next_state, done, mask = batch

    state_flipped = torch.flip(state, dims=[3])           # (batch, 3, 3, 3)
    next_state_flipped = torch.flip(next_state, dims=[3])
    prob_flipped = torch.flip(prob, dims=[1])             # (batch, 9)
    mask_flipped = torch.flip(mask, dims=[1])

    state = torch.cat([state, state_flipped], dim=0)
    next_state = torch.cat([next_state, next_state_flipped], dim=0)
    prob = torch.cat([prob, prob_flipped], dim=0)
    mask = torch.cat([mask, mask_flipped], dim=0)
    value = torch.cat([value, value], dim=0)
    done = torch.cat([done, done], dim=0)

    return state, prob, value, next_state, done, mask