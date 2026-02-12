import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_ch=1, img_size=28, latent_dim=16, beta=1.0):
        super().__init__()
        self.beta = beta
        flat = img_size*img_size*in_ch
        h = 400
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, h), nn.ReLU(),
        )
        self.mu = nn.Linear(h, latent_dim)
        self.logvar = nn.Linear(h, latent_dim)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, flat)
        )
        self.img_size, self.in_ch = img_size, in_ch

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        out = self.dec_fc(z)
        out = out.view(-1, self.in_ch, self.img_size, self.img_size)
        return torch.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def loss(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.beta*kld, {'recon':recon.detach(), 'kld':kld.detach()}
