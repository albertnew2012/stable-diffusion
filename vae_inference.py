#!/usr/bin/env python3
# vae_inference.py
# Load a trained ConvVAE checkpoint and run it on real inputs.

import os
import argparse
import torch
from torchvision import datasets, transforms, utils
from PIL import Image

from variational_autoencoder import ConvVAE, vae_loss


# ----------------------
# Loading
# ----------------------
def load_model(ckpt: str, device):
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    latent_dim = sd["fc_mu.weight"].shape[0]
    model = ConvVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, latent_dim


def one_per_digit(ds, device):
    """Pick the first test image of each class 0-9 -> [10,1,28,28]."""
    picked = {}
    for x, y in ds:
        if y not in picked:
            picked[y] = x
        if len(picked) == 10:
            break
    xs = torch.stack([picked[d] for d in range(10)])
    return xs.to(device), list(range(10))


def load_custom_image(path: str, invert: bool, device):
    """Any image file -> 28x28 grayscale tensor in MNIST convention (white ink on black)."""
    img = Image.open(path).convert("L").resize((28, 28), Image.BILINEAR)
    x = transforms.ToTensor()(img)
    if invert:
        x = 1.0 - x
    return x.unsqueeze(0).to(device), None


# ----------------------
# Modes
# ----------------------
@torch.no_grad()
def mode_reconstruct(model, x, labels, outdir, tag="reconstruct"):
    """Encode real images -> mu -> decode. Reports per-image latent code and recon error."""
    x_hat, mu, logvar, _ = model(x)
    _, bce, kld = vae_loss(x_hat, x, mu, logvar, reduction="none")

    print(f"\n{'idx':>4} {'label':>6} {'recon(BCE)':>11} {'KL':>7}   latent mu")
    print("-" * 60)
    for i in range(x.size(0)):
        lab = labels[i] if labels is not None else "-"
        z = ", ".join(f"{v:+.3f}" for v in mu[i].tolist())
        print(f"{i:>4} {str(lab):>6} {bce[i]:>11.2f} {kld[i]:>7.2f}   [{z}]")
    print("-" * 60)
    print(f"mean recon(BCE) = {bce.mean():.2f}   mean KL = {kld.mean():.2f}")

    grid = utils.make_grid(torch.cat([x, x_hat], dim=0), nrow=x.size(0), padding=2)
    path = os.path.join(outdir, f"{tag}.png")
    utils.save_image(grid, path)
    print(f"\nsaved -> {path}   (top row = input, bottom row = reconstruction)")
    return path


@torch.no_grad()
def mode_interpolate(model, x, labels, outdir, steps=12):
    """Walk the straight line in latent space between the first two inputs."""
    assert x.size(0) >= 2, "interpolate needs at least 2 inputs"
    mu, _ = model.encode(x[:2])
    a = labels[0] if labels is not None else 0
    b = labels[1] if labels is not None else 1

    alphas = torch.linspace(0, 1, steps, device=x.device).view(-1, 1)
    z_path = (1 - alphas) * mu[0:1] + alphas * mu[1:2]
    x_path = model.decode(z_path)

    grid = utils.make_grid(x_path, nrow=steps, padding=2)
    path = os.path.join(outdir, "interp_inference.png")
    utils.save_image(grid, path)
    print(f"interpolating {a} -> {b} in {steps} steps")
    print(f"saved -> {path}")
    return path


@torch.no_grad()
def mode_manifold(model, latent_dim, outdir, n=20):
    """Decode a grid over the 2-D prior: the learned digit manifold."""
    if latent_dim != 2:
        print(f"manifold needs latent_dim=2 (checkpoint has {latent_dim}) - skipping")
        return None
    # even quantiles of N(0,1), so the grid covers where the prior mass actually is
    q = torch.distributions.Normal(0.0, 1.0).icdf(torch.linspace(0.02, 0.98, n))
    z1, z2 = torch.meshgrid(q, q.flip(0), indexing="xy")
    z = torch.stack([z1.reshape(-1), z2.reshape(-1)], dim=1).to(next(model.parameters()).device)

    x_hat = model.decode(z)
    grid = utils.make_grid(x_hat, nrow=n, padding=1)
    path = os.path.join(outdir, "manifold.png")
    utils.save_image(grid, path)
    print(f"saved -> {path}   ({n}x{n} sweep over z1,z2 in [{q[0]:.2f}, {q[-1]:.2f}])")
    return path


@torch.no_grad()
def mode_sample(model, latent_dim, outdir, n=16):
    """Pure generation: draw z ~ N(0,I) and decode."""
    device = next(model.parameters()).device
    x_hat = model.decode(torch.randn(n, latent_dim, device=device))
    grid = utils.make_grid(x_hat, nrow=int(n ** 0.5), padding=2)
    path = os.path.join(outdir, "sample_inference.png")
    utils.save_image(grid, path)
    print(f"saved -> {path}")
    return path


# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run a trained ConvVAE on real inputs.")
    p.add_argument("--ckpt", type=str, default="./vae_outputs/conv_vae_mnist.pt")
    p.add_argument("--datadir", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./vae_infer_outputs")
    p.add_argument("--mode", type=str, default="reconstruct",
                   choices=["reconstruct", "interpolate", "manifold", "sample", "all"])
    p.add_argument("--image", type=str, default=None,
                   help="Run on your own image file instead of MNIST test digits")
    p.add_argument("--invert", action="store_true",
                   help="Invert --image (use for dark-ink-on-white scans)")
    p.add_argument("--digits", type=int, nargs="+", default=None,
                   help="Which digit classes to load, e.g. --digits 3 8")
    p.add_argument("--steps", type=int, default=12, help="interpolate: number of steps")
    p.add_argument("--grid", type=int, default=20, help="manifold: grid size per axis")
    p.add_argument("--n", type=int, default=16, help="sample: how many to draw")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    model, latent_dim = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt}  (latent_dim={latent_dim}, device={device})")

    # ---- meaningful input ----
    if args.image:
        x, labels = load_custom_image(args.image, args.invert, device)
    else:
        ds = datasets.MNIST(args.datadir, train=False, download=False,
                            transform=transforms.ToTensor())
        x, labels = one_per_digit(ds, device)
        if args.digits:
            sel = [labels.index(d) for d in args.digits]
            x, labels = x[sel], list(args.digits)

    if args.mode in ("reconstruct", "all"):
        mode_reconstruct(model, x, labels, args.outdir)
    if args.mode in ("interpolate", "all"):
        mode_interpolate(model, x, labels, args.outdir, steps=args.steps)
    if args.mode in ("manifold", "all"):
        mode_manifold(model, latent_dim, args.outdir, n=args.grid)
    if args.mode in ("sample", "all"):
        mode_sample(model, latent_dim, args.outdir, n=args.n)
