#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from utils import r_square


def quantile_huber_loss(pred, target, tau, kappa=1.0):
    assert pred.shape[1] == tau.shape[0], "pred and tau must have compatible shapes"
    target = target.expand_as(pred)
    diff = target - pred
    huber = torch.where(diff.abs() <= kappa, 0.5 * diff.pow(2), kappa * (diff.abs() - 0.5 * kappa))
    tau = tau.view(1, -1)
    loss = torch.abs(tau - (diff.detach() < 0).float()) * huber
    return loss.mean()


class PolicyValueNet:
    def __init__(self, net, discount, params=None):
        self.params = params
        self.net = net
        self.opt = self.net.opt
        self.discount = discount
        self.device = self.net.device
        self.n_actions = net.n_actions
        if hasattr(self.net, 'num_quantiles'):
            self.num_quantiles = net.num_quantiles
            self.tau = torch.linspace(0.5 / self.num_quantiles, 1 - 0.5 / self.num_quantiles, self.num_quantiles).to(self.device)
        else:
            self.num_quantiles = None
            self.tau = None
        if params:
            self.net.load(params)

    def policy_value(self, state):
        self.net.eval()
        with torch.no_grad():
            log_p, value_quantile = self.net(state)
            value_logit = value_quantile.mean(dim=-1)
            probs = np.exp(log_p.cpu().numpy())
            value = np.tanh(value_logit.cpu().numpy())
        return probs, value

    def policy_value_fn(self, env):
        valid = env.valid_move()
        current_state = torch.from_numpy(
            env.current_state()).float().to(self.device)
        probs, value = self.policy_value(current_state)
        action_probs = tuple(zip(valid, probs.flatten()[valid]))
        return action_probs, value.flatten()[0]

    def train_step(self, state, prob, value, mask, max_iter):
        p_l, v_l = [], []
        with torch.no_grad():
            self.eval()
            log_p, value_quantile = self.net(state)
            old_probs = np.exp(log_p.cpu().numpy())
            old_v = np.tanh(value_quantile.mean(dim=-1).cpu().numpy())
            r2_old = r_square(old_v, value)
        self.train()
        for _ in range(max_iter):
            self.opt.zero_grad()
            log_p_pred, value_quantiles = self.net(state)
            v_loss = quantile_huber_loss(torch.tanh(value_quantiles), value, self.tau)
            p_loss = F.kl_div(log_p_pred, prob, reduction='none').masked_fill_(~mask, 0).sum(dim=-1).mean()
            loss = p_loss + v_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.opt.step()
            p_l.append(p_loss.item())
            v_l.append(v_loss.item())
        self.eval()
        with torch.no_grad():
            new_probs, new_v = self.policy_value(state)
        kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-8) - np.log(new_probs + 1e-8)), axis=1))
        r2_new = r_square(new_v, value)
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.net.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), float(entropy), total_norm, kl, r2_old, r2_new

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
