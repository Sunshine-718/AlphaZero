# -*- coding: utf-8 -*-
# @Time: 2024/4/23 5:32
from environments.NBTTT.utils import print_board, create_1d_array_int, alphabeta, status, valid_movement, count_pieces, dynamic_params
import numpy as np
from copy import deepcopy


class Agent:
    def __init__(self, player, depth=None, branch=9):
        self.player = player
        self.dynamic_ = True if depth is None else False
        self.depth = depth
        self.branch = branch
        self._actions = create_1d_array_int([0 for _ in range(81)])

    # @Timer(2)
    def action(self, env: "NBTTT"):
        boards = deepcopy(env.boards.numpy()).reshape(10, 10)
        if self.dynamic_:
            self.depth, self.branch = dynamic_params(
                boards, env.count, self.depth, self.branch)
        return alphabeta(env.boards, self._actions, 0, env.curr, self.depth, self.branch, self.player)


class NBTTT:
    def __init__(self, init_board=None, init_action=None):
        self.turn = 0
        self.boards = create_1d_array_int([2 for _ in range(100)])
        self.curr = np.random.randint(
            1, 9) if init_board is None else init_board
        self.action = np.random.randint(
            1, 9) if init_action is None else init_action
        _init = self.curr
        self.step(self.curr, self.action)
        _valid = self.valid_movement()
        if _init in _valid:
            _valid.remove(_init)
        self.step(self.curr, np.random.choice(_valid))
        self.count = count_pieces(self.boards.numpy())
        self.players = ['human', 'human']

    def reset(self, config):
        players = self.players
        self.__init__()
        self.players = players
        self.players[0].depth = config['depthX']
        self.players[1].depth = config['depthO']
        return self.state()

    def join(self, playerX: None | str = None, playerO: None | str = None, config=None):
        if config is None:
            config = {'depthX': None, 'depthO': None}
        if playerX == 'human' and playerO == 'human':
            return
        if playerX != 'human':
            self.players[0] = Agent(0, config['depthX'], 9)
        if playerO != 'human':
            self.players[1] = Agent(1, config['depthO'], 9)
        infoX = f"PlayerX: {'human' if self.players[0] == 'human' else f'''AI-depth: {
            'dynamic' if config['depthX'] is None else config['depthX']}'''}"
        infoO = f"PlayerO: {'human' if self.players[1] == 'human' else f'''AI-depth: {
            'dynamic' if config['depthO'] is None else config['depthO']}'''}"
        print(infoX)
        print(infoO)

    def play(self, show=True):
        curr = self.curr
        player = self.players[self.turn]
        if show:
            self.show()
        if player == 'human':
            print(
                f"Step: {self.count + 1}:\nPlayer: {['X', 'O'][self.turn]}, curr: {self.curr}")
            print("Input an action from 1-9, q for quit: ")
            Input = input()
            if Input == 'q':
                print("Game Interrupted")
                exit()
            action = int(Input)
        else:
            action = player.action(self)
            print(
                f"Step: {self.count + 1}\nPlayer: {['X', 'O'][self.turn]}, curr: {self.curr}, action: {action}, depth: {player.depth}")
        self.step(curr, action)
        return curr, action

    def show(self):
        print('=' * 50)
        print(f'Step: {self.count}')
        print_board(self.state())
        done, winner = self.status()
        if done:
            if winner is None:
                print("Draw.")
            else:
                print(f"Player {['X', 'O'][winner]} wins!")
        print('=' * 50)

    def valid_movement(self):
        return valid_movement(self.state(), self.curr)

    def status(self):
        return status(self.state(), self.curr)

    def state(self):
        return self.boards.numpy().reshape(10, 10)

    def switch_player(self):
        self.turn = 1 - self.turn
        return self.turn

    def step(self, curr, action):
        valid = self.valid_movement()
        if action not in valid:
            print(f'Invalid action: {action}')
            return
        if self.turn == 0:
            self.boards[int(curr) * 10 + int(action)] = 0
        else:
            self.boards[int(curr) * 10 + int(action)] = 1
        self.switch_player()
        self.curr = action
        self.count = count_pieces(self.state())
        done, winner = self.status()
        if winner == 0:
            reward = [1, -1]
        elif winner == 1:
            reward = [-1, 1]
        else:
            reward = [0, 0]
        return self.state(), reward, done

    def backward(self, curr, action):
        self.boards[int(curr) * 10 + int(action)] = 2
        self.switch_player()
        self.curr = curr
        self.count = count_pieces(self.state())


class Queue:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.state = np.zeros((1, maxlen * 9, 3, 3), dtype=np.float32)
        self.idx = 0

    def add(self, state):
        self.state[:, 9:, :, :] = self.state[:, :-9, :, :]
        self.state[:, :9, :, :] = state

    def __call__(self):
        return self.state


class NBTTT_py(NBTTT):
    def __init__(self, init_board=None, init_action=None):
        self.turn = 0
        self.boards = np.array([2 for _ in range(100)])
        self.curr = np.random.randint(
            1, 9) if init_board is None else init_board
        self.action = np.random.randint(
            1, 9) if init_action is None else init_action
        _init = self.curr
        self.step(self.curr, self.action)
        _valid = self.valid_movement()
        if _init in _valid:
            _valid.remove(_init)
        self.step(self.curr, np.random.choice(_valid))
        self.count = count_pieces(self.boards.reshape(10, 10))
        self.players = ['human', 'human']

    def state(self):
        return self.boards.reshape(10, 10)

    def observation(self):
        temp = deepcopy(self.state())
        temp[np.where(temp == 1)] = -1
        temp[np.where(temp == 0)] = 1
        temp[np.where(temp == 2)] = 0
        temp = temp[1:, 1:].reshape(1, 9, 3, 3)
        return temp


if __name__ == '__main__':
    env = NBTTT_py()
    env.show()
    print(env.observation().shape)
    state = Queue(3)
    state.add(env.observation())
    print(state().shape)
    state.add(env.observation())
    print(state().shape)
    pass
