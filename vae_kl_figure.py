#!/usr/bin/env python3
# vae_kl_figure.py
# Renders figures/kl_meaning.png - what "how far is q(z|x) from the prior" means.
#
# Two distributions on one axis are two bell curves. A Gaussian has only two knobs,
# so "far" takes only two forms: wrong place (mu != 0) and wrong width (sigma != 1).
# The closed-form KL is exactly those two penalties added together.

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89"
PRIOR, POST = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
YMAX = 1.30


def kl(mu, s):
    """KL( N(mu, s^2) || N(0,1) ) for one dimension."""
    return 0.5 * (s ** 2 + mu ** 2 - 1 - 2 * np.log(s))


def npdf(z, mu, s):
    return np.exp(-0.5 * ((z - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def render(path, panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 3.9), facecolor=SURFACE)
    z = np.linspace(-4.3, 4.3, 2400)

    for ax, (mu, s, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        p, q = npdf(z, 0, 1), npdf(z, mu, s)
        ax.fill_between(z, np.minimum(p, YMAX), color=PRIOR, alpha=.15, lw=0)
        ax.plot(z, p, color=PRIOR, lw=3.4, alpha=.95, solid_capstyle="round")
        ax.fill_between(z, np.minimum(q, YMAX), color=POST, alpha=.15, lw=0)
        ax.plot(z, q, color=POST, lw=2, solid_capstyle="round")

        if q.max() > YMAX:                      # narrow spike runs off the top - say so
            ax.annotate("", xy=(mu, YMAX * 1.005), xytext=(mu, YMAX * .83),
                        arrowprops=dict(arrowstyle="-|>", color=POST, lw=1.6, mutation_scale=11))
            ax.text(mu + .28, YMAX * .90, f"peak {q.max():.1f}\n(off scale)", color=POST,
                    fontsize=8.4, fontweight="bold", va="center", linespacing=1.35)

        ax.set_ylim(0, YMAX); ax.set_xlim(-4.3, 4.3); ax.set_yticks([])
        ax.set_xticks([-3, 0, 3]); ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#dcdbd6")
        ax.grid(axis="x", color="#efeeea", lw=.8); ax.set_axisbelow(True)
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=26, loc="left")
        ax.text(0, 1.035, sub, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")
        ax.text(.5, -.30, f"KL = {kl(mu, s):.2f} nats", transform=ax.transAxes, ha="center",
                color=INK, fontsize=11.5, fontweight="bold",
                bbox=dict(fc="#f2f1ec", ec="#dcdbd6", boxstyle="round,pad=0.36"))
        ax.set_xlabel("z", color=MUTED, fontsize=9, labelpad=1)

    axes[0].annotate("prior  p(z) = N(0, I)", xy=(-1.55, npdf(-1.55, 0, 1)), xytext=(-4.15, .93),
                     color=PRIOR, fontsize=9.4, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=PRIOR, lw=1.1, shrinkA=2, shrinkB=3))
    axes[0].annotate("encoder's  q(z|x)", xy=(1.35, npdf(1.35, 0, 1)), xytext=(1.5, .72),
                     color=POST, fontsize=9.4, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=POST, lw=1.1, shrinkA=2, shrinkB=3))

    fig.suptitle('"How far is q(z|x) from the prior?"  —  KL measures the mismatch between two bell curves',
                 color=INK, fontsize=13.5, fontweight="bold", x=.006, ha="left", y=.985)
    fig.legend(handles=[plt.Line2D([], [], color=PRIOR, lw=3.4,
                                   label="prior  p(z) = N(0, I)  — fixed, never learned"),
                        plt.Line2D([], [], color=POST, lw=2,
                                   label="posterior  q(z|x)  — what the encoder claims")],
               loc="lower left", bbox_to_anchor=(.006, .002), ncol=2, frameon=False,
               fontsize=9.4, labelcolor=INK2)
    fig.tight_layout(rect=[0, .135, 1, .90]); fig.subplots_adjust(wspace=.16)
    fig.savefig(path, dpi=155, facecolor=SURFACE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Render the KL-divergence explainer figure.")
    p.add_argument("--out", type=str, default="figures/kl_meaning.png")
    p.add_argument("--ckpt", type=str, default="vae_outputs/conv_vae_mnist.pt",
                   help="Take the 4th panel's (mu, sigma) from this model's encoding of test image 0")
    args = p.parse_args()

    # default 4th panel: the real values for test image 0 (a '7'), dim 2
    mu4, s4 = 2.3296, 0.0870
    if os.path.exists(args.ckpt):
        import torch
        from torchvision import datasets, transforms
        from variational_autoencoder import ConvVAE
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if sd["fc_mu.weight"].shape[0] == 2:
            m = ConvVAE(latent_dim=2); m.load_state_dict(sd); m.eval()
            x = datasets.MNIST("./data", train=False, download=False,
                               transform=transforms.ToTensor())[0][0].unsqueeze(0)
            with torch.no_grad():
                mu, logvar = m.encode(x)
            mu4, s4 = mu[0, 1].item(), (0.5 * logvar[0, 1]).exp().item()
            print(f"panel 4 from {args.ckpt}: mu={mu4:.4f} sigma={s4:.4f}")

    panels = [(0.0, 1.00, "Identical", "q sits exactly on p — the only free case"),
              (2.0, 1.00, "Centre shifted", "same width, wrong place"),
              (0.0, 0.10, "Too narrow", "right place, over-confident"),
              (mu4, s4, "Your test '7', dim 2", "shifted AND narrow — both charged")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(args.out, panels)
    print(f"saved -> {args.out}")
    for mu, s, t, _ in panels:
        print(f"  {t:<24} mu={mu:<8.4f} sigma={s:<8.4f} KL={kl(mu, s):.4f}")
