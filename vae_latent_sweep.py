#!/usr/bin/env python3
# vae_latent_sweep.py
# Train one ConvVAE per latent_dim and compare them on the same footing.
#
# Answers: "does latent_dim matter, and where does it stop mattering?"
# Reuses ConvVAE / vae_loss from variational_autoencoder.py so the model and
# objective are identical to the main script - only latent_dim varies.

import os
import csv
import json
import math
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from variational_autoencoder import ConvVAE, vae_loss


# ----------------------
# Metrics
# ----------------------
@torch.no_grad()
def eval_elbo(model, loader, device):
    """Mean per-image reconstruction BCE and KL on the test set."""
    model.eval()
    R = K = N = 0.0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_hat, mu, logvar, _ = model(x)
        _, bce, kld = vae_loss(x_hat, x, mu, logvar, reduction="none")
        R += bce.sum().item()
        K += kld.sum().item()
        N += x.size(0)
    return R / N, K / N


@torch.no_grad()
def knn_probe(model, train_ds, test_loader, device, k=5, n_train=10000, n_test=2000):
    """
    k-NN classification on the latent means.

    The VAE never sees labels during training, so this is a PROBE: it asks
    whether same-digit images ended up near each other in latent space anyway.
    Chance is 10%; raw pixels give ~97%.
    """
    model.eval()

    def embed(loader, cap):
        Z, Y, n = [], [], 0
        for x, y in loader:
            mu, _ = model.encode(x.to(device, non_blocking=True))
            Z.append(mu)
            Y.append(y.to(device))
            n += x.size(0)
            if n >= cap:
                break
        return torch.cat(Z)[:cap], torch.cat(Y)[:cap]

    Ztr, Ytr = embed(DataLoader(train_ds, batch_size=512, num_workers=2), n_train)
    Zte, Yte = embed(test_loader, n_test)

    correct = 0
    for i in range(0, Zte.size(0), 256):
        d = torch.cdist(Zte[i:i + 256], Ztr)
        idx = d.topk(k, largest=False).indices
        vote = Ytr[idx].mode(dim=1).values
        correct += (vote == Yte[i:i + 256]).sum().item()
    return 100.0 * correct / Zte.size(0)


@torch.no_grad()
def active_units(model, loader, device, thresh=0.01):
    """
    How many latent dims actually carry information.

    A dim whose per-dim KL is ~0 has collapsed to the prior: the encoder
    ignores it and the decoder learns nothing from it. Per-dimension
    posterior collapse is the classic failure mode at high latent_dim.
    """
    model.eval()
    d = model.latent_dim
    acc = torch.zeros(d, device=device)
    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu, logvar = model.encode(x)
        acc += (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(0)
        n += x.size(0)
    per_dim = acc / n
    return int((per_dim > thresh).sum().item()), per_dim.sort(descending=True).values.tolist()


@torch.no_grad()
def save_recon_strip(model, test_ds, device, path):
    """One test digit per class 0-9: top row input, bottom row reconstruction."""
    picked = {}
    for x, y in test_ds:
        if y not in picked:
            picked[y] = x
        if len(picked) == 10:
            break
    xs = torch.stack([picked[i] for i in range(10)]).to(device)
    x_hat, _, _, _ = model(xs)
    utils.save_image(utils.make_grid(torch.cat([xs, x_hat]), nrow=10, padding=2), path)


# ----------------------
# Train one config
# ----------------------
def train_one(latent_dim, train_loader, args, device):
    torch.manual_seed(args.seed)
    model = ConvVAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    warmup_iters = int(args.kl_warmup_frac * args.epochs * math.ceil(len(train_loader)))
    it = 0
    for _ in range(args.epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            beta = min(1.0, it / warmup_iters) if warmup_iters > 0 else args.beta
            opt.zero_grad()
            x_hat, mu, logvar, _ = model(x)
            loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            it += 1
    return model


# ----------------------
# Entry
# ----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sweep latent_dim for ConvVAE on MNIST.")
    p.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    p.add_argument("--datadir", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./vae_sweep_outputs")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl_warmup_frac", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.datadir, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(args.datadir, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)

    print(f"device={device}  epochs={args.epochs}  seed={args.seed}  dims={args.dims}\n")
    rows = []
    for d in args.dims:
        model = train_one(d, train_loader, args, device)
        recon, kl = eval_elbo(model, test_loader, device)
        acc = knn_probe(model, train_ds, test_loader, device)
        alive, per_dim = active_units(model, test_loader, device)
        n_prm = sum(q.numel() for q in model.parameters())

        torch.save(model.state_dict(), os.path.join(args.outdir, f"vae_d{d:02d}.pt"))
        save_recon_strip(model, test_ds, device, os.path.join(args.outdir, f"recon_d{d:02d}.png"))

        rows.append(dict(latent_dim=d, recon_bce=round(recon, 2), kl_nats=round(kl, 2),
                         elbo=round(recon + kl, 2), knn5_acc=round(acc, 1),
                         bits=round(kl / math.log(2), 1), active_units=alive, params=n_prm,
                         per_dim_kl=[round(v, 3) for v in per_dim]))
        print(f"latent_dim={d:>3}  recon={recon:7.2f}  KL={kl:6.2f}  ELBO={recon+kl:7.2f}  "
              f"5NN={acc:5.1f}%  active={alive}/{d}  params={n_prm:,}", flush=True)

    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(dict(config=vars(args), results=rows), f, indent=2)
    with open(os.path.join(args.outdir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "per_dim_kl"])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != "per_dim_kl"})

    print(f"\n dim | recon BCE |    KL  |  ELBO  | 5NN acc | bits | active")
    print("-----+-----------+--------+--------+---------+------+-------")
    for r in rows:
        print(f" {r['latent_dim']:>3} | {r['recon_bce']:9.2f} | {r['kl_nats']:6.2f} | "
              f"{r['elbo']:6.2f} | {r['knn5_acc']:6.1f}% | {r['bits']:4.1f} | "
              f"{r['active_units']:>2}/{r['latent_dim']}")
    print(f"\nwrote checkpoints, recon strips, results.json and results.csv -> {args.outdir}/")
