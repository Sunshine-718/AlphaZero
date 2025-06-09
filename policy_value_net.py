#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:23
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.distributions import Normal, kl_divergence


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

    def policy_value(self, state, mask=None):
        self.net.eval()
        with torch.no_grad():
            # log_p, value_logit = self.net(state, mask)
            log_p, dist, *_ = self.net(state, mask)
            value_logit = dist.sample()
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
        if augmentation is not None:
            batch = augmentation(batch)
        state, prob, value, *_ = batch
        state_ = deepcopy(state)
        state_[:, -1, :, :] = -state_[:, -1, :, :]
        self.opt.zero_grad()
        log_p_pred, dist, mu, sigma = self.net(state)
        _, dist_, *_ = self.net(state_)
        # v_loss = F.smooth_l1_loss(torch.tanh(value_logit), value)
        # v_loss += F.smooth_l1_loss(torch.tanh(value_logit_), -value)
        v_loss = -dist.log_prob(value).mean()
        v_loss += -dist_.log_prob(-value).mean()
        p_loss = F.kl_div(log_p_pred, prob, reduction='batchmean')
        # 计算 KL 散度正则化项
        prior = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        kl_loss = kl_divergence(dist, prior).mean()
        loss = p_loss + v_loss + 1 * kl_loss
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
