import argparse, os, torch
from utils.data import make_loaders
from utils.seed import set_seed
from utils.early_stop import EarlyStopping
from models.vae import VAE
from models.autoencoder import AE
from train import train_epoch, eval_epoch, save_ckpt
from eval import evaluate_recon, latent_silhouette, sample_uncertainty
from torchvision.utils import save_image, make_grid
from metrics.fid_is import compute_fid_is

def get_model(name, dataset, latent_dim, beta):
    if dataset in ['MNIST','FashionMNIST']:
        in_ch, img = 1, 28
    elif dataset=='CIFAR10':
        in_ch, img = 3, 32
    else:
        raise ValueError('Unsupported dataset')
    if name=='VAE':
        return VAE(in_ch=in_ch, img_size=img, latent_dim=latent_dim, beta=beta)
    if name=='AE':
        return AE(in_ch=in_ch, img_size=img, latent_dim=latent_dim)
    raise ValueError('Unknown model')

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, default='generation', choices=['generation','evaluate','sample'])
    ap.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','FashionMNIST','CIFAR10'])
    ap.add_argument('--data-root', type=str, default='./data')
    ap.add_argument('--model', type=str, default='VAE', choices=['VAE','AE'])
    ap.add_argument('--latent-dim', type=int, default=16)
    ap.add_argument('--beta', type=float, default=4.0)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--checkpoint', type=str, default=None)
    ap.add_argument('--runs-root', type=str, default='runs')
    ap.add_argument('--mc-samples', type=int, default=16)
    ap.add_argument('--num-samples', type=int, default=64)
    return ap.parse_args()

def main():
    args = parse()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = make_loaders(
        dataset=args.dataset, data_root=args.data_root, batch_size=args.batch_size)

    model = get_model(args.model, args.dataset, args.latent_dim, args.beta).to(device)

    run_dir = os.path.join(args.runs_root, args.dataset, args.model)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, 'latest.pt')
    figs_dir = os.path.join(run_dir, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    if args.task=='generation':
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        stopper = EarlyStopping(patience=args.patience, min_delta=1e-4)
        best = float('inf')
        for epoch in range(1, args.epochs+1):
            tr = train_epoch(model, train_loader, opt, device)
            va = eval_epoch(model, val_loader, device)
            if va < best:
                best = va
                save_ckpt(model, ckpt_path)
            if stopper.step(va):
                break
        # Load best
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Save a grid of reconstructions
        x,_ = next(iter(test_loader))
        x = x.to(device)[:32]
        out = model(x)
        if len(out)==4:
            x_hat = out[0]
        else:
            x_hat = out[0]
        grid = make_grid(torch.cat([x.cpu(), x_hat.cpu()], dim=0), nrow=32)
        save_image(grid, os.path.join(figs_dir, 'recon_grid.png'))

        # Uncertainty sampling (VAE): saves variance maps
        sample_uncertainty(model, x.cpu(), device, mc_samples=args.mc_samples, out_dir=figs_dir)

        print('Training complete. Checkpoints and figures saved to:', run_dir)

    elif args.task=='evaluate':
        assert args.checkpoint, '--checkpoint required for evaluate'
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        recon_mse = evaluate_recon(model, test_loader, device)
        # FID/IS (if possible)
        def gen_fn(b):
            z = torch.randn(b, args.latent_dim, device=device)
            if hasattr(model, 'decode'):
                return model.decode(z).clamp(0,1).detach().cpu()
            else:
                return model(torch.randn_like(next(iter(test_loader))[0].to(device)))[0].clamp(0,1).cpu()
        fid_is = compute_fid_is(test_loader, gen_fn, device, num_gen=5000)
        print({'reconstruction_mse': recon_mse, **fid_is})

    elif args.task=='sample':
        assert args.checkpoint, '--checkpoint required for sample'
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        z = torch.randn(args.num_samples, args.latent_dim, device=device)
        x = model.decode(z) if hasattr(model,'decode') else None
        if x is not None:
            grid = make_grid(x.cpu(), nrow=int(args.num_samples**0.5))
            save_image(grid, os.path.join(figs_dir, 'samples.png'))
            print('Saved samples to', os.path.join(figs_dir, 'samples.png'))
        else:
            print('Model has no decode(); cannot sample.')
    else:
        raise ValueError('Unknown task')

if __name__=='__main__':
    main()
