#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    # pred: [B, N], target: [B, 1], tau: [1, N]
    error = pred.unsqueeze(2) - target.expand_as(pred).unsqueeze(1)  # [B, N, N]
    huber = torch.where(error.abs() <= kappa, 0.5 * error.pow(2), kappa * (error.abs() - 0.5 * kappa))
    loss = torch.abs(tau.unsqueeze(-1) - (error.detach() < 0).float()) * huber  # [B, N, N]
    return loss.mean()


def quantile_loss(predictions, targets, taus):
    # predictions: [batch_size, num_quantiles]
    # targets: [batch_size, 1]
    # taus: [batch_size, num_quantiles]
    targets = targets.expand_as(predictions)
    errors = targets - predictions
    loss = torch.where(errors > 0, taus * errors, (1 - taus) * (-errors))
    return loss.mean()


class PolicyValueNet:
    def __init__(self, net, params=None):
        self.params = params
        self.net = net
        self.opt = self.net.opt
        self.device = self.net.device
        self.n_actions = net.n_actions
        self.tau = torch.linspace(0, 1, net.num_quantiles).to(net.device).view(1, -1)
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
                self.opt.zero_grad()
                log_p_pred, value_pred = self.net(state)
                v_loss = quantile_loss(value_pred, value, self.tau)
                p_loss = F.kl_div(log_p_pred, prob, reduction='batchmean')
                loss = p_loss + v_loss
                loss.backward()
                self.opt.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        self.eval()
        with torch.no_grad():
            _, new_v = self.net.predict(state)
        r2 = r2_score(value.cpu().numpy(), new_v)
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
