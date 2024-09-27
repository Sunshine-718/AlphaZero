#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import torch
import torch.nn as nn
from abc import ABC


class Base(ABC, nn.Module):
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def save(self, path=None):
        if path is not None:
            self.cpu()
            torch.save(self.state_dict(), path)
            self.to(self.device)

    def load(self, path=None):
        if path is not None:
            try:
                self.cpu()
                self.load_state_dict(torch.load(path))
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
                input('Confirm to ramdomly initialize parameters.')
                self.weight_init()
            finally:
                self.to(self.device)
                