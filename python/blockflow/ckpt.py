"""Checkpoint format shared with the C/CUDA trainer (csrc/dit.cuh: dit_save / dit_load).

    int32  header[64]    magic, version, config, block id, step, flags, num_params
    double dheader[16]   t_lo, t_hi, q_lo, q_hi, logit-normal mean / std, t_scale
    float32 params[num_params]                      canonical order (model.param_specs)
    float32 ema[num_params]   if flags & FLAG_EMA
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .model import DiT, DiTConfig, num_params
from .partition import TimePartition

MAGIC, VERSION, FLAG_EMA = 4180418, 2, 1
INT_SLOTS, DBL_SLOTS = 64, 16
# int header: magic version img in_ch patch hidden depth heads mlp classes freq block nblocks step flags np_lo np_hi
# double header: t_lo t_hi q_lo q_hi ln_mean ln_std t_scale


def save_checkpoint(path: str, cfg: DiTConfig, params: torch.Tensor, *, block_id: int = 0, num_blocks: int = 1,
                    step: int = 0, ema: torch.Tensor | None = None, partition: TimePartition | None = None) -> None:
    n = num_params(cfg)
    part = partition or TimePartition(num_blocks)
    hdr = np.zeros(INT_SLOTS, dtype=np.int32)
    hdr[:17] = [MAGIC, VERSION, cfg.img_size, cfg.in_ch, cfg.patch, cfg.hidden, cfg.depth, cfg.heads, cfg.mlp_ratio,
                cfg.num_classes, cfg.freq_dim, block_id, num_blocks, step, FLAG_EMA if ema is not None else 0,
                n & 0x7FFFFFFF, n >> 31]
    dh = np.zeros(DBL_SLOTS, dtype=np.float64)
    dh[:7] = [*part.interval(block_id), *part.quantile_interval(block_id), part.mean, part.std, cfg.t_scale]
    with open(path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(dh.tobytes())
        for v in (params, ema):
            if v is not None:
                f.write(v.detach().cpu().float().numpy().tobytes())


def load_checkpoint(path: str) -> dict:
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(4 * INT_SLOTS), dtype=np.int32)
        if hdr[0] != MAGIC or hdr[1] != VERSION:
            raise ValueError(f"{path}: not a BlockFlow v{VERSION} checkpoint")
        dh = np.frombuffer(f.read(8 * DBL_SLOTS), dtype=np.float64)
        cfg = DiTConfig(img_size=int(hdr[2]), in_ch=int(hdr[3]), patch=int(hdr[4]), hidden=int(hdr[5]),
                        depth=int(hdr[6]), heads=int(hdr[7]), mlp_ratio=int(hdr[8]), num_classes=int(hdr[9]),
                        freq_dim=int(hdr[10]), t_scale=float(dh[6]))
        n = int(hdr[15]) | (int(hdr[16]) << 31)
        assert n == num_params(cfg), f"{path}: {n} parameters, expected {num_params(cfg)}"

        def vec():
            return torch.from_numpy(np.frombuffer(f.read(4 * n), dtype=np.float32).copy())

        params = vec()
        ema = vec() if hdr[14] & FLAG_EMA else None
    return dict(cfg=cfg, block_id=int(hdr[11]), num_blocks=int(hdr[12]), step=int(hdr[13]),
                ln_mean=float(dh[4]), ln_std=float(dh[5]), params=params, ema=ema)


def load_specialists(ckpt_dir: str, device: str = "cpu"):
    """Loads <dir>/block*.bin (EMA weights when present) -> (models sorted by block, partition)."""
    cks = sorted((load_checkpoint(p) for p in glob.glob(os.path.join(ckpt_dir, "block*.bin"))),
                 key=lambda c: c["block_id"])
    assert cks and [c["block_id"] for c in cks] == list(range(cks[0]["num_blocks"])), f"incomplete blocks in {ckpt_dir}"
    models = []
    for c in cks:
        m = DiT(c["cfg"]).to(device)
        m.load_flat((c["ema"] if c["ema"] is not None else c["params"]).to(device))
        models.append(m.eval())
    return models, TimePartition(len(cks), cks[0]["ln_mean"], cks[0]["ln_std"])
