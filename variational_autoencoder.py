
#!/usr/bin/env python3
# vae_mnist.py
# Convolutional Variational Autoencoder (VAE) on MNIST with visualizations.

import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple

# ----------------------
# Model
# ----------------------
class ConvVAE(nn.Module):
    """
    Convolutional VAE for 28x28 MNIST.
    Encoder: x -> (mu, logvar)
    Sampling: z = mu + std * eps
    Decoder: z -> x_hat
    """
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> 64x7x7
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),           # 7x7
            nn.ReLU(inplace=True),
        )
        self.enc_feat = 64 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_feat, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_feat, latent_dim)

        # Decoder: latent -> 64x7x7 -> 32x14x14 -> 16x28x28 -> 1x28x28
        self.fc_dec = nn.Linear(latent_dim, self.enc_feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)                      # [B,64,7,7]
        h = h.view(x.size(0), -1)            # [B, 3136]
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(z.size(0), 64, 7, 7)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


# ----------------------
# Loss (ELBO)
# ----------------------
def vae_loss(x_hat, x, mu, logvar, beta=1.0, reduction="mean"):
    # Reconstruction: BCE over pixels
    bce = F.binary_cross_entropy(x_hat, x, reduction="none")
    bce = bce.view(bce.size(0), -1).sum(dim=1)  # per-sample
    # KL(q(z|x) || N(0,I))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # per-sample
    if reduction == "mean":
        return (bce + beta * kld).mean(), bce.mean(), kld.mean()
    elif reduction == "sum":
        return (bce + beta * kld).sum(), bce.sum(), kld.sum()
    else:
        return (bce + beta * kld), bce, kld


# ----------------------
# Visualization utils
# ----------------------
def plot_loss_curves(loss_hist, recon_hist, kld_hist, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(loss_hist, label="Total")
    plt.plot(recon_hist, label="Recon")
    plt.plot(kld_hist, label="KL")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("VAE Training Losses")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

@torch.no_grad()
def visualize_reconstructions(model, loader, epoch, device, outdir, n=8):
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)[:n]
    x_hat, _, _, _ = model(x)
    grid = utils.make_grid(torch.cat([x, x_hat], dim=0), nrow=n, padding=2)
    path = os.path.join(outdir, f"reconstructions{epoch}.png")
    utils.save_image(grid, path)
    return path

@torch.no_grad()
def visualize_samples(model, device, outdir, n=16, latent_dim=2):
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    x_hat = model.decode(z)
    grid = utils.make_grid(x_hat, nrow=int(math.sqrt(n)), padding=2)
    path = os.path.join(outdir, "samples_prior.png")
    utils.save_image(grid, path)
    return path

@torch.no_grad()
def visualize_latent_scatter(model, loader, device, outdir, max_points=6000):
    model.eval()
    zs, ys = [], []
    total = 0
    for x, y in loader:
        x = x.to(device)
        mu, logvar = model.encode(x)
        z = mu  # mean as embedding
        zs.append(z.cpu())
        ys.append(y)
        total += x.size(0)
        if total >= max_points:
            break
    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()
    plt.figure(figsize=(5,5))
    plt.scatter(Z[:,0], Z[:,1], c=Y, s=6, cmap="tab10", alpha=0.75)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title("Latent (μ) Scatter — colored by digit")
    plt.tight_layout()
    path = os.path.join(outdir, "latent_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

@torch.no_grad()
def visualize_interpolation(model, loader, device, outdir, steps=10):
    """
    Pick two images, encode → μ, interpolate linearly in latent space, decode grid.
    """
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    # pick two different digits
    idx_a = 0
    idx_b = (y != y[idx_a]).nonzero(as_tuple=False)[0].item() if (y != y[idx_a]).any() else min(1, len(y)-1)
    xa, xb = x[idx_a:idx_a+1], x[idx_b:idx_b+1]
    mu_a, _ = model.encode(xa)
    mu_b, _ = model.encode(xb)

    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    z_path = (1 - alphas) * mu_a + alphas * mu_b
    x_path = model.decode(z_path)
    grid = utils.make_grid(x_path, nrow=steps, padding=2)
    path = os.path.join(outdir, "interpolation.png")
    utils.save_image(grid, path)
    return path

# ----------------------
# Training
# ----------------------
def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)

    # Data
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.datadir, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.datadir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model/optim
    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optional: KL annealing (linear warmup over first N epochs)
    total_iters = args.epochs * math.ceil(len(train_loader))
    warmup_iters = int(args.kl_warmup_frac * total_iters)

    loss_hist, recon_hist, kld_hist = [], [], []
    it = 0

    for epoch in range(args.epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)

            # Linear KL weight schedule
            if warmup_iters > 0:
                beta = min(1.0, it / warmup_iters)
            else:
                beta = args.beta

            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            loss, rec, kld = vae_loss(x_hat, x, mu, logvar, beta=beta, reduction="mean")
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())
            recon_hist.append(rec.item())
            kld_hist.append(kld.item())
            it += 1

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"loss={loss_hist[-1]:.4f}  recon={recon_hist[-1]:.4f}  kl={kld_hist[-1]:.4f}  beta={beta:.3f}")

        # quick reconstruction snapshot each epoch
        visualize_reconstructions(model, test_loader,epoch, device, args.outdir, n=8)

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(args.outdir, "conv_vae_mnist.pt"))
    loss_path = plot_loss_curves(loss_hist, recon_hist, kld_hist, args.outdir)
    # recon_path = visualize_reconstructions(model, test_loader, device, args.outdir, n=8)
    sample_path = visualize_samples(model, device, args.outdir, n=16, latent_dim=args.latent_dim)

    latent_path = None
    if args.latent_dim == 2:
        latent_path = visualize_latent_scatter(model, test_loader, device, args.outdir, max_points=6000)
        interp_path = visualize_interpolation(model, test_loader, device, args.outdir, steps=12)
    else:
        interp_path = None

    print("\nArtifacts:")
    print("  Model checkpoint:", os.path.join(args.outdir, "conv_vae_mnist.pt"))
    print("  Loss curves:     ", loss_path)
    print("  Recon grid:      ", recon_path)
    print("  Samples (prior): ", sample_path)
    if latent_path:
        print("  Latent scatter:  ", latent_path)
    if interp_path:
        print("  Interpolation:   ", interp_path)

# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default="./vae_outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=2, help="Use 2 for latent scatter + interpolation")
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight (if no warmup)")
    parser.add_argument("--kl_warmup_frac", type=float, default=0.3, help="Fraction of iters to anneal KL from 0→1")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args)
