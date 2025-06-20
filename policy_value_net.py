#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


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

    def policy_value(self, state, mask=None):
        self.net.eval()
        with torch.no_grad():
            log_p, value_quantile = self.net(state, mask)
            value_logit = value_quantile.mean(dim=-1)
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

    def train_step(self, batch, augmentation=None):
        self.net.train()
        batch = augmentation(batch)
        state, prob, value, *_ = batch
        state_ = deepcopy(state)
        state_[:, -1, :, :] = -state_[:, -1, :, :]
        self.opt.zero_grad()
        log_p_pred, value_quantiles = self.net(state)
        _, value_quantiles_ = self.net(state_)
        v_loss = quantile_huber_loss(torch.tanh(value_quantiles), value, self.tau)
        v_loss += quantile_huber_loss(torch.tanh(value_quantiles_), -value, self.tau)
        p_loss = -torch.sum(prob * log_p_pred, dim=1).mean()
        loss = p_loss + v_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
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

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, env):
        return self.policy_value_fn(env)
