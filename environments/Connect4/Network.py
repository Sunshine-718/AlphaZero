#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from einops import rearrange
from ..NetworkBase import Base


class CNN(Base):
    def __init__(self, lr, in_dim=3, h_dim=128, out_dim=7, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 3), padding=(2, 2), bias=True),
                                    nn.BatchNorm2d(h_dim),
                                    nn.SiLU(True),
                                    nn.Conv2d(h_dim, h_dim * 2,
                                              kernel_size=(3, 4), bias=True),
                                    nn.BatchNorm2d(h_dim * 2),
                                    nn.SiLU(True),
                                    nn.Conv2d(h_dim * 2, h_dim * 4,
                                              kernel_size=(3, 3), bias=True),
                                    nn.BatchNorm2d(h_dim * 4),
                                    nn.SiLU(True),
                                    nn.Conv2d(h_dim * 4, h_dim * 8,
                                              kernel_size=(4, 4), bias=True),
                                    nn.BatchNorm2d(h_dim * 8),
                                    nn.SiLU(True),
                                    nn.Flatten())
        self.policy_head = nn.Sequential(nn.Linear(h_dim * 8, h_dim * 4, bias=True),
                                         nn.BatchNorm1d(h_dim * 4),
                                         nn.SiLU(True),
                                         nn.Linear(
                                             h_dim * 4, h_dim * 4, bias=True),
                                         nn.BatchNorm1d(h_dim * 4),
                                         nn.SiLU(True),
                                         nn.Linear(h_dim * 4, out_dim))
        self.value_head = nn.Sequential(nn.Linear(h_dim * 8, h_dim * 4, bias=True),
                                        nn.BatchNorm1d(h_dim * 4),
                                        nn.SiLU(True),
                                        nn.Linear(
                                            h_dim * 4, h_dim * 4, bias=True),
                                        nn.BatchNorm1d(h_dim * 4),
                                        nn.SiLU(True),
                                        nn.Linear(h_dim * 4, 1))
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        self.device = device
        self.n_actions = out_dim
        self.opt = AdamW(self.parameters(), lr=lr, weight_decay=0.1)
        self.weight_init()
        self.to(self.device)

    def name(self):
        return 'CNN'

    def forward(self, x, mask=None):
        hidden = self.hidden(x)
        prob_logit = self.policy_head(hidden)
        if mask is not None:
            prob_logit.masked_fill_(~mask, -float('inf'))
        log_prob = F.log_softmax(prob_logit, dim=-1)
        return log_prob, self.value_head(hidden)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(1, 1))
        num_patches = 6 * 7
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim))
        nn.init.orthogonal_(self.pos_embed)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, 'n c h w -> n (h w) c')
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embed
        return x


class ViT(Base):
    def __init__(self, lr, in_channels=3, embed_dim=64, num_action=7, depth=6, num_heads=8, dropout=0.1, device='cpu'):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, activation=nn.SiLU(True), norm_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth)
        self.policy_head = nn.Sequential(nn.BatchNorm1d(embed_dim),
                                         nn.Linear(embed_dim, num_action))
        self.value_head = nn.Sequential(nn.BatchNorm1d(embed_dim),
                                        nn.Linear(embed_dim, embed_dim),
                                        nn.SiLU(True),
                                        nn.Linear(embed_dim, 1))
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.zeros_(self.policy_head[-1].bias)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        self.n_actions = num_action
        self.device = device
        self.to(device)
        self.opt = AdamW(self.parameters(), lr, weight_decay=0.1)

    def name(self):
        return 'ViT'

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        x = self.transformer(x)
        cls_token = x[:, 0, :]
        prob_logit = self.policy_head(cls_token)
        if mask is not None:
            prob_logit.masked_fill_(~mask, -float('inf'))
        log_prob = F.log_softmax(prob_logit, dim=-1)
        value_logit = self.value_head(cls_token)
        return log_prob, value_logit
