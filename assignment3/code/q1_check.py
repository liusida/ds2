import numpy as np
import torch
torch.manual_seed(0)

# sanity check
x = torch.rand([5,10])
w = torch.rand([10,1])
y = torch.rand([5])

w.requires_grad_(True)

y_hat = torch.mm(x,w).flatten()
l = (y-y_hat)**2
l = l.sum()
l.backward()
print(w.grad)

with torch.no_grad():
    m = -2 * torch.mm(x.T, torch.unsqueeze(y - y_hat,-1))
    print(m)
