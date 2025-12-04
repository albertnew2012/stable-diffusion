#!/usr/bin/env python3
"""
tiny_diffusion_cifar.py
denoising diffusion probabilistic models
A minimal DDPM-style diffusion model on CIFAR-10.
This is a tiny, unconditional “Stable Diffusion core”:
- forward noising schedule
- UNet epsilon-predictor
- reverse sampling

No VAE, no text encoder, but algorithmically the same backbone.
"""

import os
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# -----------------------------
# 1. Beta schedule & utilities
# -----------------------------
def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

class DiffusionConfig:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device
        self.betas = make_beta_schedule(T, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        """
        q(x_t | x_0) = sqrt(bar_alpha_t) * x_0 + sqrt(1 - bar_alpha_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_at = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_at = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_at * x0 + sqrt_one_minus_at * noise

# -----------------------------
# 2. Tiny UNet backbone
# -----------------------------
class ResidBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, out_ch)
            )

        self.block1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.block2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.block1(x)
        if (self.time_mlp is not None) and (t_emb is not None):
            # add time embedding as bias
            h = h + self.time_mlp(t_emb).view(h.size(0), -1, 1, 1)
        h = self.act(h)
        h = self.block2(h)
        h = h + self.res_conv(x)
        h = self.act(h)
        return h

class TinyUNet(nn.Module):
    """
    Very small UNet for 32x32 images.
    """
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.enc1 = ResidBlock(in_ch, base_ch, time_emb_dim)          # 3  -> 64
        self.enc2 = ResidBlock(base_ch, base_ch * 2, time_emb_dim)    # 64 -> 128
        self.enc3 = ResidBlock(base_ch * 2, base_ch * 4, time_emb_dim) # 128 -> 256

        self.down = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ResidBlock(base_ch * 4, base_ch * 4, time_emb_dim)  # 256 -> 256

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Decoder
        # d3 input: up(mid): 256, concat h2: 128 => 256+128 = 384
        self.dec3 = ResidBlock(base_ch * 4 + base_ch * 2, base_ch * 2, time_emb_dim)  # 384 -> 128

        # d2 input: up(d3): 128, concat h1: 64 => 128+64 = 192
        self.dec2 = ResidBlock(base_ch * 2 + base_ch, base_ch, time_emb_dim)          # 192 -> 64

        # d1 input: concat d2 (64) with x (3) => 67
        self.dec1 = ResidBlock(base_ch + in_ch, base_ch, time_emb_dim)                # 67 -> 64

        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)

    def pos_encoding(self, t, scale=1000):
        # t: [B] integers in [0, T)
        # Map to [0, 1] then reshape to [B,1]
        return (t.float() / scale).unsqueeze(-1)

    def forward(self, x, t):
        """
        x: [B, 3, 32, 32], t: [B] timesteps
        Predicts epsilon (noise) for DDPM objective.
        """
        t = self.pos_encoding(t)        # [B, 1]
        t_emb = self.time_mlp(t)        # [B, time_emb_dim]

        # Encoder
        h1 = self.enc1(x, t_emb)                    # [B, 64, 32, 32]
        h2 = self.enc2(self.down(h1), t_emb)        # [B, 128, 16, 16]
        h3 = self.enc3(self.down(h2), t_emb)        # [B, 256, 8, 8]

        # Bottleneck
        mid = self.mid(h3, t_emb)                   # [B, 256, 8, 8]

        # Decoder
        d3 = self.up(mid)                           # [B, 256, 16, 16]
        d3 = torch.cat([d3, h2], dim=1)             # [B, 256+128=384, 16, 16]
        d3 = self.dec3(d3, t_emb)                   # [B, 128, 16, 16]

        d2 = self.up(d3)                            # [B, 128, 32, 32]
        d2 = torch.cat([d2, h1], dim=1)             # [B, 128+64=192, 32, 32]
        d2 = self.dec2(d2, t_emb)                   # [B, 64, 32, 32]

        d1 = torch.cat([d2, x], dim=1)              # [B, 64+3=67, 32, 32]
        d1 = self.dec1(d1, t_emb)                   # [B, 64, 32, 32]

        out = self.out_conv(d1)                     # [B, 3, 32, 32]
        return out

# -----------------------------
# 3. Training & sampling
# -----------------------------
def ddpm_loss(model, diffusion, x0):
    """
    Sample random t and noise, compute MSE between predicted eps and true eps.
    """
    b = x0.size(0)
    t = torch.randint(0, diffusion.T, (b,), device=x0.device)
    eps = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, eps)
    eps_pred = model(x_t, t)
    return F.mse_loss(eps_pred, eps)

@torch.no_grad()
def sample(model, diffusion, n_samples=16, img_size=32, channels=3):
    model.eval()
    device = diffusion.device
    x_t = torch.randn(n_samples, channels, img_size, img_size, device=device)
    for t in reversed(range(diffusion.T)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)
        beta_t = diffusion.betas[t]
        alpha_t = diffusion.alphas[t]
        alpha_bar_t = diffusion.alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alphas_cumprod[t]

        # DDPM sampling step:
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_pred) + sigma_t * z
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / sqrt_one_minus_alpha_bar
        x0_pred = (x_t - coef2 * eps_pred) * coef1  # for intuition, though not used directly

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(diffusion.posterior_variance[t])
            x_t = coef1 * (x_t - coef2 * eps_pred) + sigma_t * noise
        else:
            x_t = coef1 * (x_t - coef2 * eps_pred)
    return x_t

def plot_loss(loss_hist, outdir):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.title("Diffusion training loss")
    plt.tight_layout()
    path = os.path.join(outdir, "loss_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

# -----------------------------
# 4. Main training script
# -----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # Data: CIFAR-10, scaled to [-1,1] or [0,1]? Here use [-1,1] for variety.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)  # map from [0,1] -> [-1,1]
    ])
    ds = datasets.CIFAR10(root=args.datadir, train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    diffusion = DiffusionConfig(T=args.T, device=device)
    model = TinyUNet(in_ch=3, base_ch=args.base_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_hist = []
    it = 0
    for epoch in range(args.epochs):
        for x, _ in dl:
            x = x.to(device)
            opt.zero_grad()
            loss = ddpm_loss(model, diffusion, x)
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())
            it += 1
            if it % args.log_interval == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] it={it} loss={loss.item():.4f}")

            if it >= args.max_iters:
                break
        if it >= args.max_iters:
            break

    ckpt_path = os.path.join(args.outdir, "tiny_diffusion.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint:", ckpt_path)

    loss_path = plot_loss(loss_hist, args.outdir)
    print("Saved loss curve:", loss_path)

    # sample some images
    samples = sample(model, diffusion, n_samples=16, img_size=32, channels=3)
    # map from [-1,1] back to [0,1]
    samples = (samples.clamp(-1,1) + 1) / 2
    grid = utils.make_grid(samples, nrow=4)
    img_path = os.path.join(args.outdir, "samples.png")
    utils.save_image(grid, img_path)
    print("Saved samples:", img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default="./tiny_diffusion_outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200, help="Number of diffusion timesteps (smaller for speed)")
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=2000, help="Stop early for demo")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    train(args)
