#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class PolicyValueNet:
    def __init__(self, net, discount, params=None):
        self.params = params
        self.net = net
        self.opt = self.net.opt
        self.discount = discount
        self.device = self.net.device
        if params:
            self.net.load(params)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, env):
        return self.policy_value_fn(env)

    def policy_value(self, state):
        self.net.eval()
        with torch.no_grad():
            log_p, value = self.net(state)
            probs = np.exp(log_p.cpu().numpy())
        return probs, value.cpu().numpy()

    def policy_value_fn(self, env):
        self.net.eval()
        valid = env.valid_move()
        current_state = torch.from_numpy(
            env.current_state()).float().to(self.device)
        probs, value = self.policy_value(current_state)
        action_probs = list(zip(valid, probs.flatten()[valid]))
        return action_probs, value.flatten()[0]

    def train_step(self, batch):
        self.net.train()
        state, prob, value, _ = batch
        oppo_state = deepcopy(state)
        oppo_state[:, -1, :, :] = -oppo_state[:, -1, :, :]
        self.opt.zero_grad()
        log_p_pred, value_pred = self.net(state)
        _, oppo_value_pred = self.net(oppo_state)
        v_loss = F.smooth_l1_loss(value_pred, value)
        v_loss += F.smooth_l1_loss(oppo_value_pred, -value)
        entropy = -torch.mean(torch.sum(torch.exp(log_p_pred) * log_p_pred, 1))
        p_loss = F.kl_div(log_p_pred, prob, reduction='batchmean')
        loss = p_loss + v_loss
        loss.backward()
        total_norm = 0
        for param in self.net.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.opt.step()
        self.net.eval()
        return p_loss.item(), v_loss.item(), entropy.item(), total_norm

    def save(self, params=None):
        if params is None:
            params = self.params
        self.net.save(params)

    def load(self, params=None):
        if params is None:
            params = self.params
        self.net.load(params)
