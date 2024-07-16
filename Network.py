#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class Network(nn.Module):
    def __init__(self, lr, in_dim, h_dim, out_dim, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 3), padding=(2, 2)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim, h_dim * 2,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 2, h_dim * 4,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(h_dim * 4, h_dim * 8,
                                              kernel_size=(3, 3)),
                                    nn.LeakyReLU(0.2, True),)
        self.policy = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Flatten(),
                                    nn.Linear(h_dim * 4, h_dim * 4),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Linear(h_dim * 4, out_dim),
                                    nn.LogSoftmax(dim=1))
        self.value = nn.Sequential(nn.Conv2d(h_dim * 8, h_dim * 4, kernel_size=(2, 3)),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Flatten(),
                                   nn.Linear(h_dim * 4, h_dim * 4),
                                   nn.LeakyReLU(0.2, True),
                                   nn.Linear(h_dim * 4, 1),
                                   nn.Tanh())
        self.device = device
        self.opt = Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.weight_init()
        self.to(self.device)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
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


class PolicyValueNet:
    def __init__(self, lr, params=None, device='cpu'):
        self.device = device
        self.params = params
        self.policy_value_net = torch.compile(Network(lr, 3, 64, 7, device), mode='max-autotune')
        self.opt = self.policy_value_net.opt
        if params:
            self.policy_value_net.load(params)

    def policy_value(self, state):
        self.policy_value_net.eval()
        with torch.no_grad():
            log_p, value = self.policy_value_net(state)
            probs = np.exp(log_p.cpu().numpy())
        return probs, value.cpu().numpy()

    def policy_value_fn(self, env):
        self.policy_value_net.eval()
        valid = env.valid_move()
        current_state = np.ascontiguousarray(env.current_state())
        with torch.no_grad():
            log_p, value = self.policy_value_net(
                torch.from_numpy(current_state).float().to(self.device))
            probs = np.exp(log_p.cpu().numpy().flatten())
            value = value.item()
        action_probs = list(zip(valid, probs[valid]))
        return action_probs, value

    def train_step(self, batch):
        self.policy_value_net.train()
        criterion = nn.KLDivLoss(reduction="batchmean")
        for state, prob, value in batch:
            self.opt.zero_grad()
            log_p_pred, value_pred = self.policy_value_net(state)
            v_loss = F.smooth_l1_loss(value_pred, value)
            # p_loss = -torch.mean(torch.sum(prob * log_p_pred, 1))
            p_loss = criterion(log_p_pred, prob)
            loss = p_loss + v_loss
            loss.backward()
            self.opt.step()
            entropy = -torch.mean(torch.sum(torch.exp(log_p_pred) * log_p_pred, 1))
        return loss.item(), entropy.item()

    def save(self, params=None):
        if params is None:
            params = self.params
        self.policy_value_net.save(params)
    
    def load(self, params=None):
        if params is None:
            params = self.params
        self.policy_value_net.load(params)


if __name__ == '__main__':

    pass
