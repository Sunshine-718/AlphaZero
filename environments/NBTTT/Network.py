#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 15/Sep/2024  17:23
import torch
import torch.nn as nn
from torch.optim import NAdam
from ..NetworkBase import Base


class CNN(Base):
    def __init__(self, lr, in_dim, h_dim, out_dim, device='cpu'):
        super().__init__()
        self.hidden1 = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(1, 3), bias=False),
                                     nn.BatchNorm2d(h_dim),
                                     nn.SiLU(True),
                                     nn.Conv2d(h_dim, h_dim * 2, kernel_size=(3, 1), bias=False),
                                     nn.BatchNorm2d(h_dim * 2),
                                     nn.SiLU(True),
                                     nn.Flatten())
        self.hidden2 = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 1), bias=False),
                                     nn.BatchNorm2d(h_dim),
                                     nn.SiLU(True),
                                     nn.Conv2d(h_dim, h_dim * 2, kernel_size=(1, 3), bias=False),
                                     nn.BatchNorm2d(h_dim * 2),
                                     nn.SiLU(True),
                                     nn.Flatten())
        self.hidden3 = nn.Sequential(nn.Conv2d(in_dim, h_dim, kernel_size=(3, 3), padding=(2, 2), bias=False),
                                     nn.BatchNorm2d(h_dim),
                                     nn.SiLU(True),
                                     nn.Conv2d(h_dim, h_dim * 2, kernel_size=(3, 3), bias=False),
                                     nn.BatchNorm2d(h_dim * 2),
                                     nn.SiLU(True),
                                     nn.Conv2d(h_dim * 2, h_dim * 4, kernel_size=(3, 3), bias=False),
                                     nn.BatchNorm2d(h_dim * 4),
                                     nn.SiLU(True),
                                     nn.Flatten())
        self.policy_head = nn.Sequential(nn.Linear(h_dim * 8, h_dim * 4, bias=False),
                                         nn.BatchNorm1d(h_dim * 4),
                                         nn.SiLU(True),
                                         nn.Linear(h_dim * 4, h_dim * 4, bias=False),
                                         nn.BatchNorm1d(h_dim * 4),
                                         nn.SiLU(True),
                                         nn.Linear(h_dim * 4, out_dim),
                                         nn.LogSoftmax(dim=1))
        self.value_head = nn.Sequential(nn.Linear(h_dim * 8, h_dim * 4, bias=False),
                                        nn.BatchNorm1d(h_dim * 4),
                                        nn.SiLU(True),
                                        nn.Linear(h_dim * 4, h_dim * 4, bias=False),
                                        nn.BatchNorm1d(h_dim * 4),
                                        nn.SiLU(True),
                                        nn.Linear(h_dim * 4, 1),
                                        nn.Tanh())
        self.device = device
        self.n_actions = out_dim
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=0.1, decoupled_weight_decay=True)
        self.weight_init()
        self.to(self.device)

    def forward(self, x):
        hidden = torch.concat([self.hidden1(x), self.hidden2(x), self.hidden3(x)], dim=1)
        return self.policy_head(hidden), self.value_head(hidden)
