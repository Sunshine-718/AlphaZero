#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Written by: Sunshine
# Created on: 20/Sep/2024  17:46
import torch

def instant_augment(batch):
    states, probs, values, _ = batch
    states = torch.concat([states, -states])
    probs = torch.concat([probs, torch.fliplr(probs)])
    values = torch.concat([values, values])
    return states, probs, values, None

if __name__ == '__main__':
    
    pass
