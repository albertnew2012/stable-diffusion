#!/usr/bin/env python3
# gan_mnist.py
# Simple convolutional GAN on MNIST with visualizations.

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
# Models
# ----------------------
class Generator(nn.Module):
    """
    DCGAN-style generator for 28x28 MNIST.
    Input: z ~ N(0,1), shape [B, z_dim]
    Output: fake image in [0,1], shape [B,1,28,28]
    """
    def __init__(self, z_dim=64):
        super().__init__()
        self.z_dim = z_dim
        # Project and reshape to 128 x 7 x 7
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        # Upsample: 7x7 -> 14x14 -> 28x28
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # keep in [0,1] to match MNIST ToTensor
        )

    def forward(self, z):
        h = self.fc(z)                         # [B, 128*7*7]
        h = h.view(z.size(0), 128, 7, 7)       # [B, 128, 7, 7]
        x_fake = self.net(h)                   # [B, 1, 28, 28]
        return x_fake


class Discriminator(nn.Module):
    """
    CNN-based discriminator for 28x28 MNIST.
    Input: image [B,1,28,28]
    Output: logit [B,1] (no sigmoid; use BCEWithLogitsLoss).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 7x7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.net(x)                        # [B,128,7,7]
        h = h.view(x.size(0), -1)              # [B, 128*7*7]
        logit = self.fc(h)                     # [B,1]
        return logit


# ----------------------
# Visualization utils
# ----------------------
def plot_loss_curves(d_losses, g_losses, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(d_losses, label="D loss")
    plt.plot(g_losses, label="G loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("GAN Training Losses")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "gan_loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


@torch.no_grad()
def visualize_samples(generator, fixed_z, device, outdir, epoch=None, nrow=8):
    generator.eval()
    z = fixed_z.to(device)
    x_fake = generator(z)
    grid = utils.make_grid(x_fake, nrow=nrow, padding=2)
    if epoch is None:
        fname = "gan_samples_final.png"
    else:
        fname = f"gan_samples_epoch{epoch:03d}.png"
    path = os.path.join(outdir, fname)
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

    # Data: MNIST in [0,1]
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.datadir, train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    # Models
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss: use BCEWithLogits (D outputs logits)
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for visualization
    fixed_z = torch.randn(args.sample_grid_size, args.z_dim)

    d_losses, g_losses = [], []
    it = 0

    for epoch in range(args.epochs):
        G.train()
        D.train()
        for x_real, _ in train_loader:
            x_real = x_real.to(device)
            batch_size = x_real.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --------------------
            # 1) Update Discriminator
            # --------------------
            opt_D.zero_grad()

            # D(real)
            logits_real = D(x_real)
            d_loss_real = criterion(logits_real, real_labels)

            # D(fake) with G(z) (detach so gradients don't go to G)
            z = torch.randn(batch_size, args.z_dim, device=device)
            x_fake = G(z).detach()
            logits_fake = D(x_fake)
            d_loss_fake = criterion(logits_fake, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()

            # --------------------
            # 2) Update Generator
            # --------------------
            opt_G.zero_grad()

            z = torch.randn(batch_size, args.z_dim, device=device)
            x_fake = G(z)
            logits_fake = D(x_fake)
            # Generator tries to make D think these are real (label=1)
            g_loss = criterion(logits_fake, real_labels)
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            it += 1

        print(f"[Epoch {epoch+1}/{args.epochs}] "
              f"D_loss={d_losses[-1]:.4f}  G_loss={g_losses[-1]:.4f}")

        # Snapshot samples each epoch using fixed noise
        visualize_samples(G, fixed_z, device, args.outdir, epoch=epoch+1, nrow=int(math.sqrt(args.sample_grid_size)))

    # Save artifacts
    torch.save(G.state_dict(), os.path.join(args.outdir, "gan_generator.pt"))
    torch.save(D.state_dict(), os.path.join(args.outdir, "gan_discriminator.pt"))
    loss_path = plot_loss_curves(d_losses, g_losses, args.outdir)
    final_sample_path = visualize_samples(G, fixed_z, device, args.outdir, epoch=None,
                                          nrow=int(math.sqrt(args.sample_grid_size)))

    print("\nArtifacts:")
    print("  Generator checkpoint:   ", os.path.join(args.outdir, "gan_generator.pt"))
    print("  Discriminator checkpoint:", os.path.join(args.outdir, "gan_discriminator.pt"))
    print("  Loss curves:            ", loss_path)
    print("  Final sample grid:      ", final_sample_path)


# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default="./gan_outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--sample_grid_size", type=int, default=64,
                        help="Number of samples in the visualization grid (must be a square like 16,25,36,64)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(args)
