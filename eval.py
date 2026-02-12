import torch, numpy as np, os
from torchvision.utils import save_image, make_grid
from sklearn.metrics import silhouette_score
from tqdm import tqdm

def evaluate_recon(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            if hasattr(model, 'loss'):
                if len(out)==4:
                    x_hat, mu, logvar, z = out
                    loss, parts = model.loss(x, x_hat, mu, logvar)
                    losses.append(parts['recon'].item())
                else:
                    x_hat, z = out
                    loss = model.loss(x, x_hat)
                    losses.append(loss.item())
            else:
                raise ValueError('Model must implement loss')
    return float(np.mean(losses))

def latent_silhouette(model, loader, device):
    # Optional analysis: encode to z and run silhouette over labels (if available) or pseudo-labels via kmeans
    from sklearn.cluster import KMeans
    model.eval()
    Z = []
    Y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if len(out)==4:
                _, mu, _, z = out
                z_feat = mu
            else:
                _, z_feat = out
            Z.append(z_feat.detach().cpu().numpy())
            Y.append(y.numpy())
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)
    # If labels are digits (MNIST), silhouette by true labels is illustrative; else do kmeans
    try:
        score_true = silhouette_score(Z, Y)
    except Exception:
        score_true = None
    k = len(np.unique(Y)) if len(np.unique(Y))>1 else 10
    km = KMeans(n_clusters=k, n_init='auto').fit(Z)
    score_km = silhouette_score(Z, km.labels_)
    return {'silhouette_true': (float(score_true) if score_true is not None else None),
            'silhouette_kmeans': float(score_km)}

def sample_uncertainty(model, x, device, mc_samples=16, out_dir=None, prefix='unc'):
    # Draw multiple samples per input to estimate pixel-wise variance (for VAE)
    model.eval()
    x = x.to(device)
    imgs = []
    with torch.no_grad():
        if hasattr(model, 'encode'):
            mu, logvar = model.encode(x)
            for _ in range(mc_samples):
                z = model.reparameterize(mu, logvar)
                imgs.append(model.decode(z).detach().cpu())
        else:
            # AE deterministic: single reconstruction (no uncertainty)
            imgs.append(model(x)[0].detach().cpu())
    stack = torch.stack(imgs, dim=0)  # [S, B, C, H, W]
    var = stack.var(dim=0, unbiased=False)  # [B, C, H, W]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        grid = make_grid(stack[:,0], nrow=mc_samples)
        save_image(grid, os.path.join(out_dir, f'{prefix}_samples.png'))
        save_image(var[0], os.path.join(out_dir, f'{prefix}_var.png'))
    return stack, var
