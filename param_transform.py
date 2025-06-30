import torch
import torch.nn as nn
from environments.Connect4.Network import CNN


net = CNN(0)
net.load('./params/CNN_pretrained.pt')
