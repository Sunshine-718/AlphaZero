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
        self.n_actions = net.n_actions
        if params:
            self.net.load(params)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, env):
        return self.policy_value_fn(env)

    def policy_value(self, state, mask=None):
        self.net.eval()
        with torch.no_grad():
            log_p, value_logit = self.net(state, mask)
            probs = np.exp(log_p.cpu().numpy())
            value = np.tanh(value_logit.cpu().numpy())
        return probs, value

    def policy_value_fn(self, env):
        self.net.eval()
        valid = env.valid_move()
        current_state = torch.from_numpy(
            env.current_state()).float().to(self.device)
        mask = torch.tensor(env.valid_mask(), dtype=torch.bool, device=self.device).unsqueeze(0)
        probs, value = self.policy_value(current_state, mask)
        action_probs = list(zip(valid, probs.flatten()[valid]))
        return action_probs, value.flatten()[0]

    def train_step(self, batch):
        self.net.train()
        state, prob, value, _, _, mask = batch
        oppo_state = deepcopy(state)
        oppo_state[:, -1, :, :] = -oppo_state[:, -1, :, :]
        self.opt.zero_grad()
        log_p_pred, value_logit = self.net(state)
        _, oppo_value_pred = self.net(oppo_state)
        v_loss = F.binary_cross_entropy_with_logits(value_logit, (value + 1) * 0.5)
        v_loss += F.binary_cross_entropy_with_logits(oppo_value_pred, (-value + 1) * 0.5)
        p_loss = (F.kl_div(log_p_pred, prob, reduction='none') * mask).mean()
        loss = p_loss + v_loss
        loss.backward()
        self.opt.step()
        self.net.eval()
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_p_pred) * log_p_pred, 1))
            total_norm = 0
            for param in self.net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return float(p_loss), float(v_loss), float(entropy), total_norm

    def save(self, params=None):
        if params is None:
            params = self.params
        self.net.save(params)

    def load(self, params=None):
        if params is None:
            params = self.params
        self.net.load(params)
