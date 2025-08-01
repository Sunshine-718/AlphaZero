#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam
from einops import rearrange
from ..NetworkBase import Base
from .config import network_config as config


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(out_dim),
                                  nn.SiLU(True),
                                  nn.Dropout2d(0.1),
                                  nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(out_dim))
    
    def forward(self, x):
        return F.silu(x + self.conv(x))


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=config['h_dim'], out_dim=7, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(True),
                                    nn.Dropout2d(0.1),
                                    Block(h_dim, h_dim),
                                    Block(h_dim, h_dim),
                                    Block(h_dim, h_dim))
        self.policy_head = nn.Sequential(nn.Conv2d(h_dim, 3, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(3),
                                         nn.SiLU(True),
                                         nn.Flatten(),
                                         nn.Linear(3 * 6 * 7, out_dim),
                                         nn.LogSoftmax(dim=-1))
        self.value_head = nn.Sequential(nn.Conv2d(h_dim, 1, kernel_size=1, bias=False),
                                         nn.BatchNorm2d(1),
                                         nn.SiLU(True),
                                         nn.Flatten(),
                                         nn.Linear(6 * 7, 3),
                                         nn.LogSoftmax(dim=-1))
        self.device = device
        self.n_actions = out_dim
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=1e-4, decoupled_weight_decay=True)
        self.to(self.device)

    def name(self):
        return 'CNN'

    def forward(self, x):
        hidden = self.hidden(x)
        log_prob = self.policy_head(hidden)
        value = self.value_head(hidden)
        return log_prob, value
    
    def policy(self, state):
        self.eval()
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            return self.policy_head(self.hidden(state)).exp().cpu().numpy()
    
    def value(self, state):
        self.eval()
        with torch.no_grad():
            return self.value_head(self.hidden(state)).exp().cpu().numpy()
    
    def predict(self, state, draw_factor=0.5):
        self.eval()
        with torch.no_grad():
            log_prob, value_log_prob = self.forward(state)
            value_prob = value_log_prob.exp()
            player = state[:, -1, 0, 0].view(-1,)
            value = (-player) * draw_factor * value_prob[:, 0] + value_prob[:, 1] - value_prob[:, 2]
        return log_prob.exp().cpu().numpy(), value.cpu().view(-1, 1).numpy()


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))
        num_patches = 6 * 7
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim))
        self.d_model = embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'n c h w -> n (h w) c')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) * math.sqrt(self.d_model)
        x += self.pos_embed
        return x


class ViT(Base):
    def __init__(self, lr, in_channels=3, embed_dim=128, num_action=7, depth=4, num_heads=8, dropout=0.1, device='cpu'):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation=nn.SiLU(True), norm_first=False, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.policy_head = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                         nn.LayerNorm(embed_dim),
                                         nn.SiLU(True),
                                         nn.Linear(embed_dim, num_action))
        self.value_head = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                        nn.LayerNorm(embed_dim),
                                        nn.SiLU(True),
                                        nn.Linear(embed_dim, 1))
        self.n_actions = num_action
        self.device = device
        self.to(device)
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=1e-4, decoupled_weight_decay=True)

    def name(self):
        return 'ViT'

    def forward(self, x, mask=None):
        cls_token = self.get_cls_token(x)
        prob_logit = self.policy_head(cls_token)
        if mask is not None:
            prob_logit.masked_fill_(~mask, -float('inf'))
        log_prob = F.log_softmax(prob_logit, dim=-1)
        value_logit = self.value_head(cls_token)
        return log_prob, value_logit
    
    def get_cls_token(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        return x[:, 0, :]
