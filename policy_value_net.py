#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from copy import deepcopy


class PolicyValueNet:
    def __init__(self, net, params=None):
        self.params = params
        self.net = net
        self.opt = self.net.opt
        self.device = self.net.device
        self.n_actions = net.n_actions
        if params:
            self.net.load(params)
        self.eval()

    def policy_value(self, state):
        return self.net.predict(state)

    def policy_value_fn(self, env):
        valid = env.valid_move()
        current_state = torch.from_numpy(
            env.current_state()).float().to(self.device)
        probs, value = self.policy_value(current_state)
        action_probs = tuple(zip(valid, probs.flatten()[valid]))
        return action_probs, value.flatten()[0]

    def train_step(self, dataloader, augment, current_step):
        p_l, v_l = [], []
        self.train()
        for _ in range(3):
            for batch in dataloader:
                state, prob, value, *_ = augment(batch)
                value = deepcopy(value)
                value[value == -1] = 2
                value = value.view(-1, )
                self.opt.zero_grad()
                log_p_pred, value_pred = self.net(state)
                v_loss = F.nll_loss(value_pred, value.type(torch.int64))
                p_loss = torch.mean(torch.sum(-prob * log_p_pred, dim=1))
                loss = p_loss + v_loss
                loss.backward()
                self.opt.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        self.eval()
        with torch.no_grad():
            _, new_v = self.net(state)
        r2 = f1_score(value.cpu().numpy(), torch.argmax(new_v, dim=-1).cpu().numpy(), average='macro')
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), float(entropy), total_norm, r2

    def save(self, params=None):
        if params is None:
            params = self.params
        self.net.save(params)

    def load(self, params=None):
        if params is None:
            params = self.params
        self.net.load(params)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, env):
        return self.policy_value_fn(env)
