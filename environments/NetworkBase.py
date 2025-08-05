#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC
from copy import deepcopy
from sklearn.metrics import f1_score


class Base(ABC, nn.Module):
    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is not None:
            try:
                self.load_state_dict(torch.load(path, map_location=self.device))
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
    
    def train_step(self, dataloader, augment):
        p_l, v_l = [], []
        self.train()
        for _ in range(3):
            for batch in dataloader:
                state, prob, _, winner, *_ = augment(batch)
                value = deepcopy(winner)
                value[value == -1] = 2
                value = value.view(-1,).long()
                value_oppo = deepcopy(winner)
                value_oppo[value_oppo == 1] = -2
                value_oppo = (-value_oppo).view(-1,).long()
                state_oppo = deepcopy(state)
                state_oppo[:, -1, :, :] = -state_oppo[:, -1, :, :]
                self.opt.zero_grad()
                log_p_pred, value_pred = self(state)
                _, value_oppo_pred = self(state_oppo)
                v_loss = F.nll_loss(value_pred, value)
                v_loss += F.nll_loss(value_oppo_pred, value_oppo)
                p_loss = torch.mean(torch.sum(-prob * log_p_pred, dim=1))
                loss = p_loss + v_loss
                loss.backward()
                self.opt.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        self.eval()
        with torch.no_grad():
            _, new_v = self(state)
        f1 = f1_score(value.cpu().numpy(), torch.argmax(new_v, dim=-1).cpu().numpy(), average='macro')
        with torch.no_grad():
            entropy = -torch.mean(torch.sum(log_p_pred.exp() * log_p_pred, dim=-1))
            total_norm = 0
            for param in self.parameters():
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return np.mean(p_l), np.mean(v_l), float(entropy), total_norm, f1
                