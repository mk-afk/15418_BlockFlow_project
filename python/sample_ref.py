"""Samples from block checkpoints (C or PyTorch trained) with the PyTorch reference sampler.

    python python/sample_ref.py --ckpt runs/b2 --n 64 --grid grid.png

EMA weights, 50 Euler steps, labels cycling through the classes. --npy saves uint8 (N, H, W, C).
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockflow import euler_sample, load_specialists  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--grid", default=None)
    ap.add_argument("--npy", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    models, part = load_specialists(a.ckpt, a.device)
    cfg = models[0].cfg
    g = torch.Generator().manual_seed(0)
    out = []
    for start in range(0, a.n, 256):
        m = min(256, a.n - start)
        x1 = torch.randn(m, cfg.in_ch, cfg.img_size, cfg.img_size, generator=g).to(a.device)
        y = (torch.arange(start, start + m) % cfg.num_classes).to(a.device)
        x = euler_sample(models, part, x1, y, 50)
        out.append(((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    images = np.concatenate(out)
    if a.npy:
        np.save(a.npy, images)
    if a.grid:
        n, h, w, c = images[:64].shape
        cols, rows = 8, (n + 7) // 8
        grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        for i, im in enumerate(images[:64]):
            grid[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
        Image.fromarray(grid.squeeze(-1) if c == 1 else grid).resize((cols * w * 2, rows * h * 2), Image.NEAREST).save(a.grid)
    print(f"sampled {len(images)} images from {len(models)} block(s)")


if __name__ == "__main__":
    main()
