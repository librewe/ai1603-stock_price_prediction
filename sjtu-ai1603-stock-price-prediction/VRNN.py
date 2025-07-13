# VRNN.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- Encoder q(z_t | x_t, h_{t-1}) -----------------
class InferenceNetwork(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.fc_mu     = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=-1)
        h   = self.fc(inp)
        mu  = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# --------------- Prior p(z_t | h_{t-1}) ------------------------
class PriorNetwork(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.fc_mu     = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, h):
        h = self.fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# --------------- Decoder p(x_t | z_t, h_{t-1}) -----------------
class GenerationNetwork(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.ReLU()
        )
        self.fc_out = nn.Linear(h_dim, x_dim)

    def forward(self, z, h):
        inp = torch.cat([z, h], dim=-1)
        h = self.fc(inp)
        return self.fc_out(h)

# --------------- VRNN Cell ------------------------------------
class VRNNCell(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim):
        super().__init__()
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

        self.inference = InferenceNetwork(h_dim, h_dim, z_dim)
        self.prior     = PriorNetwork(h_dim, z_dim)
        self.generation = GenerationNetwork(z_dim, h_dim, x_dim)

        self.rnn = nn.GRUCell(h_dim + h_dim, h_dim)

    def forward(self, x, h):
        """
        x : [batch, x_dim]
        h : [batch, h_dim]
        returns: recon_x, kl_div, h_next
        """
        x_ = self.phi_x(x)

        # --- inference ---
        q_mu, q_logvar = self.inference(x_, h)
        z = self.reparameterize(q_mu, q_logvar)

        # --- prior ---
        p_mu, p_logvar = self.prior(h)

        # --- generation ---
        z_ = self.phi_z(z)
        recon_x = self.generation(z_, h)

        # --- RNN update ---
        h_next = self.rnn(torch.cat([x_, z_], dim=-1), h)

        # --- KL divergence ---
        kl_div = -0.5 * torch.sum(1 + q_logvar - p_logvar
                                  - (q_logvar.exp() + (q_mu - p_mu).pow(2)) / p_logvar.exp(),
                                  dim=-1).mean()
        return recon_x, kl_div, h_next

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# --------------- Full VRNN model -------------------------------
class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers=1):
        super().__init__()
        self.cell = VRNNCell(x_dim, h_dim, z_dim)

    def forward(self, x):
        """
        x: [batch, seq_len, x_dim]
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.cell.rnn.hidden_size, device=x.device)
        recon_loss = 0.
        kl_loss = 0.

        for t in range(seq_len):
            recon_x, kl, h = self.cell(x[:, t], h)
            recon_loss += F.mse_loss(recon_x, x[:, t], reduction='mean')
            kl_loss += kl

        return recon_loss / seq_len, kl_loss / seq_len