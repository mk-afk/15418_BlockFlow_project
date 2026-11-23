"""PyTorch reference trainer (one process): DiT-S/2, same hyperparameters, data, log and checkpoint
format as the C/CUDA trainer. Used for the reference MNIST samples and the speed comparison.

    python python/train_ref.py --data data/mnist_train.bin --batch 128 --steps 3000 --out runs/mnist_pytorch
"""
import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockflow import DiT, DiTConfig, TimePartition, patchify, save_checkpoint  # noqa: E402
from blockflow.data import read_dataset  # noqa: E402

LR, BETAS, EPS, WEIGHT_DECAY, EMA_MAX_DECAY = 1e-4, (0.9, 0.999), 1e-8, 0.0, 0.9999  # as csrc/trainer.cuh
LOGNORM_MEAN, LOGNORM_STD, DEPTH, LOG_EVERY = 0.0, 1.0, 12, 50


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512, help="images per block per step")
    ap.add_argument("--micro", type=int, default=32, help="images per micro-batch (about 135 MB each: 32 fit an 8 GB GPU)")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out", default="runs/ref")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev, B = torch.device(a.device), a.blocks
    micro, accum = a.micro, a.batch // a.micro
    assert accum * micro == a.batch, "--batch must be a multiple of --micro"

    imgs, labels, ncls = read_dataset(a.data)
    images, labels = torch.from_numpy(imgs).to(dev), torch.from_numpy(labels.astype("int64")).to(dev)
    cfg = DiTConfig(img_size=images.shape[-1], in_ch=images.shape[1], num_classes=ncls, depth=DEPTH // B)
    part = TimePartition(B, LOGNORM_MEAN, LOGNORM_STD)
    blocks = []
    for b in range(B):
        torch.manual_seed(b)
        model = DiT(cfg).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)
        blocks.append(dict(model=model, ema=copy.deepcopy(model).requires_grad_(False), opt=opt, loss=0.0))
    gen = torch.Generator().manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "log.jsonl"), "w")

    t0 = t_log = time.time()
    last = 0
    for step in range(a.steps):
        for b, blk in enumerate(blocks):
            blk["opt"].zero_grad(set_to_none=True)
            for _ in range(accum):
                idx = torch.randint(0, images.shape[0], (micro,), generator=gen).to(dev)
                x0 = images[idx].float() / 127.5 - 1.0
                t = part.sample_t(b, micro, generator=gen).to(dev)
                noise = torch.randn(x0.shape, generator=gen).to(dev)
                tt = t.view(-1, 1, 1, 1)
                xp, target = patchify((1 - tt) * x0 + tt * noise, cfg.patch), patchify(noise - x0, cfg.patch)
                loss = F.mse_loss(blk["model"].forward_tokens(xp, t, labels[idx]), target)
                (loss / accum).backward()
                blk["loss"] += loss.item() / accum
            blk["opt"].step()
            decay = min(EMA_MAX_DECAY, (2 + step) / (11 + step))  # (1 + s) / (10 + s) after s = step + 1 updates
            with torch.no_grad():
                for pe, pm in zip(blk["ema"].parameters(), blk["model"].parameters()):
                    pe.mul_(decay).add_(pm, alpha=1 - decay)
        if (step + 1) % LOG_EVERY == 0 or step + 1 == a.steps:
            if dev.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            ms = 1000 * (now - t_log) / (step + 1 - last)
            losses = [blk["loss"] / (step + 1 - last) for blk in blocks]
            for blk in blocks:
                blk["loss"] = 0.0
            print(f"step {step + 1:6d} | {ms:8.1f} ms/step | loss " + " ".join(f"b{b} {l:.4f}" for b, l in enumerate(losses)))
            log.write(json.dumps(dict(step=step + 1, time_s=now - t0, ms_per_step=ms, loss=losses)) + "\n")
            log.flush()
            t_log, last = now, step + 1
    for b, blk in enumerate(blocks):
        save_checkpoint(os.path.join(a.out, f"block{b}.bin"), cfg, blk["model"].flat_params(), block_id=b,
                        num_blocks=B, step=a.steps, ema=blk["ema"].flat_params(), partition=part)


if __name__ == "__main__":
    main()
