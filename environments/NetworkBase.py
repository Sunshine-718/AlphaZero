#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 09/Sep/2024  04:20
import torch
import torch.nn as nn
from abc import ABC


class Base(ABC, nn.Module):
    def save(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)

    def load(self, path=None):
        if path is not None:
            try:
                self.load_state_dict(torch.load(path, map_location=self.device))
                print('Load parameters successfully!')
            except Exception as e:
                print(f'Failed to load parameters.\n{e}')
                