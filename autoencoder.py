import torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):
    def __init__(self, in_ch=1, img_size=28, latent_dim=16):
        super().__init__()
        flat = img_size*img_size*in_ch
        h = 400
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, h), nn.ReLU(),
            nn.Linear(h, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h), nn.ReLU(),
            nn.Linear(h, flat)
        )
        self.img_size, self.in_ch = img_size, in_ch

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z).view(-1, self.in_ch, self.img_size, self.img_size)
        return x_hat, z

    def loss(self, x, x_hat):
        return F.mse_loss(x_hat, x, reduction='mean')
