import torch


a = torch.zeros((3, 3, 3, 3))
b = a[:, 0] + a[:, 1]
n = [torch.sum(torch.logical_not((i == 1).all(dim=0)), None, True) for i in b]
n = torch.concat(n)
print(-torch.mean(torch.log(n + 1e-8)))