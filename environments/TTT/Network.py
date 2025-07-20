import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import NAdam
from ..NetworkBase import Base


class CNN(Base):
    def __init__(self, lr=1e-3, in_dim=3, h_dim=128, out_dim=9, num_quantiles=51, device='cpu'):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(in_dim * 3 * 3, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True),
                                    nn.Linear(h_dim, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True),
                                    nn.Linear(h_dim, h_dim),
                                    nn.BatchNorm1d(h_dim),
                                    nn.SiLU(True))
        self.policy_head = nn.Linear(h_dim, out_dim)
        self.value_head = nn.Sequential(nn.Linear(h_dim, num_quantiles),
                                        nn.Tanh())
        self.device = device
        self.n_actions = out_dim
        self.num_quantiles = num_quantiles
        self.opt = NAdam(self.parameters(), lr=lr, weight_decay=0.01, decoupled_weight_decay=True)
        self.to(device)

    def name(self):
        return "CNN"

    def forward(self, x, mask=None):
        x = x.view(-1, 27)
        hidden = self.hidden(x)
        prob_logit = self.policy_head(hidden)
        if mask is not None:
            prob_logit.masked_fill_(~mask, -float('inf'))
        log_prob = F.log_softmax(prob_logit, dim=-1)
        value = self.value_head(hidden)
        return log_prob, value
    
    def predict(self, state, mask=None):
        self.eval()
        with torch.no_grad():
            log_prob, value = self.forward(state, mask)
            value = value.mean(dim=-1, keepdim=True)
        return log_prob.exp().cpu().numpy(), value.cpu().numpy()
