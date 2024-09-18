#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 14/Jul/2024  20:41
import torch.nn as nn
from torch.optim import NAdam
from ..NetworkBase import Base


class Network(Base):
    def __init__(self, lr, in_dim, h_dim, out_dim, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True),
                                    nn.Linear(h_dim, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True),
                                    nn.Linear(h_dim, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True))
        self.policy_head = nn.Sequential(nn.Linear(h_dim, h_dim),
                                         nn.BatchNorm1d(h_dim),
                                         nn.SiLU(True),
                                         nn.Linear(h_dim, h_dim),
                                         nn.BatchNorm1d(h_dim),
                                         nn.SiLU(True),
                                         nn.Linear(h_dim, out_dim),
                                         nn.LogSoftmax(dim=1))
        self.value_head = nn.Sequential(nn.Linear(h_dim, h_dim),
                                        nn.BatchNorm1d(h_dim),
                                        nn.SiLU(True),
                                        nn.Linear(h_dim, h_dim),
                                        nn.BatchNorm1d(h_dim),
                                        nn.SiLU(True),
                                        nn.Linear(h_dim, 1))
        self.device = device
        self.n_actions = out_dim
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=1e-4, decoupled_weight_decay=True)
        self.weight_init()
        self.to(self.device)

    def forward(self, x):
        hidden = self.hidden(x)
        return self.policy_head(hidden), self.value_head(hidden)
