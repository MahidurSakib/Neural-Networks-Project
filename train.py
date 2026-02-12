import torch, os
from tqdm import tqdm

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        if len(out)==4:
            x_hat, mu, logvar, z = out
            loss, _ = model.loss(x, x_hat, mu, logvar)
        else:
            x_hat, z = out
            loss = model.loss(x, x_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            if len(out)==4:
                x_hat, mu, logvar, z = out
                loss, _ = model.loss(x, x_hat, mu, logvar)
            else:
                x_hat, z = out
                loss = model.loss(x, x_hat)
            total += loss.item()
    return total/len(loader)

def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
