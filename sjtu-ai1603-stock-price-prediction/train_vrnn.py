# train_vrnn.py
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from VRNN import VRNN

# 人造正弦波
seq_len = 50
x = torch.sin(torch.linspace(0, 100, 2000)).reshape(-1, seq_len, 1).float()
dataset = TensorDataset(x)
loader  = DataLoader(dataset, batch_size=32, shuffle=True)

model = VRNN(x_dim=1, h_dim=64, z_dim=16)#%
opt   = torch.optim.Adam(model.parameters(), 1e-3)

for epoch in range(50):
    for (xb,) in loader:
        xb = xb#%
        recon, kl = model(xb)
        loss = recon + kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"epoch {epoch:02d}  recon={recon.item():.4f}  kl={kl.item():.4f}")