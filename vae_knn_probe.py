#!/usr/bin/env python3
# vae_knn_probe.py
# k-NN probe on VAE latent codes: is the latent space semantically organized?
#
# The VAE trains WITHOUT labels - vae_loss only ever sees pixels. This script asks
# whether same-digit images ended up near each other anyway, purely as a side effect
# of "reconstruct well while staying near N(0,I)".
#
# It is a PROBE: a diagnostic bolted on after training, not part of the model.
#
#   1. encode N train images -> mu vectors, keep their labels
#   2. encode a test image   -> mu
#   3. find the k closest train codes by Euclidean distance
#   4. majority vote of their labels = prediction
#
# Chance is 10%. Raw pixels give ~97%.

import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from variational_autoencoder import ConvVAE


def load_model(ckpt, device):
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    model = ConvVAE(latent_dim=sd["fc_mu.weight"].shape[0]).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def embed(model, ds, cap, device, batch_size=512):
    """Encode up to `cap` images to their latent means. Returns (Z, labels, indices)."""
    Z, Y, n = [], [], 0
    for x, y in DataLoader(ds, batch_size=batch_size, num_workers=2):
        mu, _ = model.encode(x.to(device, non_blocking=True))
        Z.append(mu)
        Y.append(y.to(device))
        n += x.size(0)
        if n >= cap:
            break
    return torch.cat(Z)[:cap], torch.cat(Y)[:cap]


@torch.no_grad()
def accuracy(Zte, Yte, Ztr, Ytr, k):
    """Overall k-NN accuracy, chunked so the distance matrix stays small."""
    correct = 0
    for i in range(0, Zte.size(0), 256):
        d = torch.cdist(Zte[i:i + 256], Ztr)
        idx = d.topk(k, largest=False).indices
        correct += (Ytr[idx].mode(dim=1).values == Yte[i:i + 256]).sum().item()
    return 100.0 * correct / Zte.size(0)


@torch.no_grad()
def inspect(Zte, Yte, Ztr, Ytr, probe, k):
    """Neighbours of a single test image. Returns (true, vote, labels, dists, train_idx)."""
    d = torch.cdist(Zte[probe:probe + 1], Ztr)[0]
    dist, idx = d.topk(k, largest=False)
    return (Yte[probe].item(), Ytr[idx].mode().values.item(),
            Ytr[idx].tolist(), dist.tolist(), idx.tolist())


def strip(test_ds, train_ds, probe, train_idx, path):
    """Save [query | k neighbours] as one row so you can eyeball what it matched."""
    imgs = [test_ds[probe][0]] + [train_ds[i][0] for i in train_idx]
    utils.save_image(utils.make_grid(torch.stack(imgs), nrow=len(imgs), padding=2), path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="k-NN probe on VAE latent codes.")
    p.add_argument("--ckpt", type=str, nargs="+",
                   default=["vae_sweep_outputs/vae_d02.pt", "vae_sweep_outputs/vae_d16.pt"],
                   help="One or more checkpoints to compare")
    p.add_argument("--probe", type=int, nargs="+", default=[8, 0],
                   help="Test-set indices to inspect in detail (8 is a '5', 0 is a '7')")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n_train", type=int, default=10000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--datadir", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./vae_sweep_outputs")
    p.add_argument("--no_images", action="store_true", help="Skip writing neighbour strips")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(args.datadir, train=True, download=False, transform=tfm)
    test_ds = datasets.MNIST(args.datadir, train=False, download=False, transform=tfm)

    for ckpt in args.ckpt:
        model = load_model(ckpt, device)
        d = model.latent_dim
        Ztr, Ytr = embed(model, train_ds, args.n_train, device)
        Zte, Yte = embed(model, test_ds, args.n_test, device)

        print(f"\n===== {ckpt}  (latent_dim={d}) =====")
        for probe in args.probe:
            true, vote, labels, dists, tidx = inspect(Zte, Yte, Ztr, Ytr, probe, args.k)
            print(f"  test img #{probe}: TRUE label = {true}")
            print(f"    {args.k} nearest training codes -> labels {labels}   "
                  f"distances {[round(t, 3) for t in dists]}")
            print(f"    majority vote = {vote}   -> {'CORRECT' if vote == true else 'WRONG'}")
            if not args.no_images:
                tag = os.path.splitext(os.path.basename(ckpt))[0]
                path = os.path.join(args.outdir, f"knn_{tag}_probe{probe}.png")
                strip(test_ds, train_ds, probe, tidx, path)
                print(f"    saved -> {path}   (leftmost = query, then the {args.k} neighbours)")

        print(f"  ---> over all {args.n_test} test images: "
              f"{accuracy(Zte, Yte, Ztr, Ytr, args.k):.1f}% correct")

    print("\nNOTE: distances are NOT comparable across latent_dim - more dimensions means "
          "larger absolute distances. Only the neighbour LABELS matter.")
