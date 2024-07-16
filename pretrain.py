#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 13/Jul/2024  17:06
from env import Env, board_to_state
from MCTS import MCTSNode, MCTS_policy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class Network(nn.Module):
    def __init__(self, lr, in_dim, h_dim, out_dim, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 3), padding=(2, 2)),
                                    nn.BatchNorm2d(h_dim),
                                    nn.Dropout2d(0.1),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim, h_dim * 2,
                                              kernel_size=(3, 3)),
                                    nn.BatchNorm2d(h_dim * 2),
                                    nn.Dropout2d(0.1),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 2, h_dim * 4,
                                              kernel_size=(3, 3)),
                                    nn.BatchNorm2d(h_dim * 4),
                                    nn.Dropout2d(0.1),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 4, h_dim * 8,
                                              kernel_size=(3, 3)),
                                    nn.BatchNorm2d(h_dim * 8),
                                    nn.Dropout2d(0.2),
                                    nn.LeakyReLU(0.2, True),)
        self.policy = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Flatten(),
                                    nn.Linear(h_dim * 4, h_dim * 4),
                                    nn.Tanh(),
                                    nn.Linear(h_dim * 4, out_dim),
                                    nn.LogSoftmax(dim=1))
        self.value = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Flatten(),
                                   nn.Linear(h_dim * 4, h_dim * 4),
                                   nn.Tanh(),
                                   nn.Linear(h_dim * 4, 1),
                                   nn.Tanh())
        self.device = device
        self.opt = Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.weight_init()
        self.to(self.device)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is not None:
            try:
                self.load_state_dict(torch.load(path))
            except Exception as e:
                print(f'failed to load parameters\n{e}')

    def forward(self, x):
        hidden = self.hidden(x)
        return self.policy(hidden), self.value(hidden)

if __name__ == '__main__':
    states = None
    actions = None
    winners = None
    env = Env()
    num_game = 1000
    net = Network(1e-3, 3, 32, 7, 'cuda')
    params = './params/pretrained.pt'
    net.load(params)
    criterion = nn.NLLLoss()
    Loss = []
    Entropy = []
    plt.ion()
    for i in tqdm(range(num_game)):
        state = env.reset()
        node = MCTSNode(env, board_to_state(state, env.turn))
        done = False
        # env.show()
        init_step = 2000
        temp_state = []
        temp_action = []
        while not done:
            temp_state.append(node.state)
            next_node, action = MCTS_policy(
                node, 1, 1, init_step, show_iter=0)
            temp_action.append(action)
            # for i in node.valid_action():
            #     print(node.child[i])
            _, _, done, _ = env.step(action)
            # env.show()
            node = next_node
        winner = env.winPlayer()
        temp_winner = torch.zeros((len(temp_state))).float().cuda()
        for i in range(len(temp_winner)):
            if i % 2 == 0:
                temp_winner[i] = winner
            else:
                temp_winner[i] = -winner
        if states is None:
            states = torch.from_numpy(np.concatenate(
                temp_state, axis=0)).float().cuda()
            actions = torch.FloatTensor(temp_action).float().cuda()
            winners = temp_winner
        else:
            states = torch.concat([states, torch.from_numpy(
                np.concatenate(temp_state, axis=0)).float().cuda()], dim=0)
            actions = torch.concat(
                [actions, torch.FloatTensor(temp_action).float().cuda()])
            winners = torch.concat([winners, temp_winner])
        dataset = TensorDataset(
            states, actions.reshape(-1, 1), winners.reshape(-1, 1))
        dataset = DataLoader(dataset, batch_size=128, shuffle=True)
        temp_loss = []
        temp_entropy = []
        net.train()
        for state, action, value in dataset:
            p_h, v_h = net(state)
            p_loss = criterion(p_h, action.reshape(-1, ).long())
            v_loss = F.smooth_l1_loss(v_h, value)
            loss = p_loss + v_loss
            p = F.softmax(p_h, dim=1)
            temp_entropy.append(-torch.mean(torch.sum(p *
                                torch.exp(p), 1)).item())
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()
            temp_loss.append(loss.item())
        net.eval()
        Entropy.append(np.mean(temp_entropy))
        Loss.append(np.mean(temp_loss))
        net.save(params)
        plt.clf()
        plt.subplot(121)
        plt.plot(Loss)
        plt.title(f'p_loss: {p_loss: .5f}\nv_loss: {v_loss: .5f}')
        plt.subplot(122)
        plt.plot(Entropy)
        plt.tight_layout()
        plt.pause(0.1)
        pass
