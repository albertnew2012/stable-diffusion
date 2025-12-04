#!/usr/bin/env python3
# ae_mnist.py
# Standard (deterministic) convolutional Autoencoder on MNIST with visualizations.

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

# ----------------------
# Model
# ----------------------
class ConvAE(nn.Module):
    """
    Convolutional Autoencoder.
    Encoder compresses to a 2-D latent for easy visualization.
    """
    def __init__(self, latent_dim=2):
        super().__init__()
        # Encoder: 1x28x28 -> 32x14x14 -> 64x7x7 -> 128x4x4 -> flatten -> FC -> latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# 4x4 (since 7->4 via stride=2,pad=1)
            nn.ReLU(inplace=True),
        )
        self.enc_out_hw = (4, 4)
        self.enc_out_ch = 128
        self.flatten_dim = self.enc_out_ch * self.enc_out_hw[0] * self.enc_out_hw[1]
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder: latent_dim -> FC -> 128x4x4 -> upsample -> 64x7x7 -> upsample -> 32x14x14 -> upsample -> 1x28x28
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> but start at 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),   # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),   # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=2),  # keep size ~32x32 then crop to 28x28
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)                        # [B, 128, 4, 4]
        h = h.view(x.size(0), -1)              # [B, 2048]
        z = self.fc_mu(h)                      # [B, latent_dim]
        return z

    def decode(self, z):
        h = self.fc_dec(z)                     # [B, 2048]
        h = h.view(z.size(0), self.enc_out_ch, *self.enc_out_hw)  # [B,128,4,4]
        x_hat_big = self.dec(h)                # [B,1,~32,~32]
        # center-crop to 28x28
        _, _, H, W = x_hat_big.shape
        sH = (H - 28) // 2
        sW = (W - 28) // 2
        x_hat = x_hat_big[:, :, sH:sH+28, sW:sW+28]
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

# ----------------------
# Utils
# ----------------------
def plot_loss_curve(losses, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("BCE Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    path = os.path.join(outdir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def visualize_reconstructions(model, loader, device, outdir, n=8):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)[:n]
    with torch.no_grad():
        recons, _ = model(imgs)
    # make a grid: first row originals, second row reconstructions
    grid = utils.make_grid(torch.cat([imgs, recons], dim=0), nrow=n, padding=2)
    path = os.path.join(outdir, "reconstructions.png")
    utils.save_image(grid, path)
    return path

def visualize_latent(model, loader, device, outdir, max_points=5000):
    model.eval()
    zs = []
    ys = []
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            z = model.encode(x)  # [B, 2]
            zs.append(z.cpu())
            ys.append(y)
            total += x.size(0)
            if total >= max_points:
                break
    Z = torch.cat(zs, dim=0).numpy()
    Y = torch.cat(ys, dim=0).numpy()
    plt.figure(figsize=(5,5))
    scatter = plt.scatter(Z[:,0], Z[:,1], c=Y, s=6, cmap="tab10", alpha=0.7)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("2D Latent Space (color by label)")
    plt.tight_layout()
    path = os.path.join(outdir, "latent_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# ----------------------
# Training
# ----------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.outdir, exist_ok=True)

    # Data
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.datadir, train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(args.datadir, train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model/optim
    model = ConvAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    losses = []
    model.train()
    it = 0
    for epoch in range(args.epochs):
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat, _ = model(x)
            loss = F.binary_cross_entropy(x_hat, x, reduction="mean")
            loss.backward()
            opt.step()

            losses.append(loss.item())
            it += 1

        print(f"[Epoch {epoch+1}/{args.epochs}] loss={losses[-1]:.4f}")

        # quick reconstruction snapshot each epoch
        visualize_reconstructions(model, test_loader, device, args.outdir, n=8)

    # Save artifacts
    torch.save(model.state_dict(), os.path.join(args.outdir, "conv_ae_mnist.pt"))
    loss_path = plot_loss_curve(losses, args.outdir)
    recon_path = visualize_reconstructions(model, test_loader, device, args.outdir, n=8)
    latent_path = None
    if args.latent_dim == 2:
        latent_path = visualize_latent(model, test_loader, device, args.outdir, max_points=5000)

    print("\nArtifacts:")
    print("  Model checkpoint:", os.path.join(args.outdir, "conv_ae_mnist.pt"))
    print("  Loss curve:     ", loss_path)
    print("  Recon grid:     ", recon_path)
    if latent_path:
        print("  Latent scatter: ", latent_path)

# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default="./ae_outputs")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=2, help="Use 2 for latent scatter plot")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    train(args)
