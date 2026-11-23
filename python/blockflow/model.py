"""PyTorch reference DiT: the correctness oracle for the C/CUDA trainer.

Class-conditional DiT (Peebles & Xie, 2023) with adaLN-Zero conditioning and a flow-matching
velocity objective. One DiT is one specialist: the end-to-end model has depth 12, a B-block model
is B independent DiTs of depth 12 / B, each with its own embeddings and output layer.

Parameters are stacked tensors in the canonical order of `param_specs`. The C/CUDA trainer, the
checkpoints and the gradient all-reduce use the same flat order.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN_EPS = 1e-6


@dataclass
class DiTConfig:
    """Defaults are DiT-S/2 on 32x32 images (32.5 M parameters at depth 12)."""
    img_size: int = 32
    in_ch: int = 3
    depth: int = 12
    patch: int = 2
    hidden: int = 384
    heads: int = 6
    num_classes: int = 10
    mlp_ratio: int = 4
    freq_dim: int = 256
    t_scale: float = 1000.0  # t in [0, 1] is scaled before the sinusoidal embedding

    @property
    def grid(self) -> int:
        return self.img_size // self.patch

    @property
    def tokens(self) -> int:
        return self.grid * self.grid

    @property
    def patch_dim(self) -> int:
        return self.patch * self.patch * self.in_ch


def param_specs(cfg: DiTConfig) -> list[tuple[str, tuple[int, ...]]]:
    """Canonical flat parameter order. Must match csrc/dit.cuh::fill_param_sizes."""
    C, P, Fd, L, NC = cfg.hidden, cfg.patch_dim, cfg.freq_dim, cfg.depth, cfg.num_classes
    H = cfg.mlp_ratio * C
    return [
        ("x_emb_w", (C, P)), ("x_emb_b", (C,)),
        ("t_w1", (C, Fd)), ("t_b1", (C,)),
        ("t_w2", (C, C)), ("t_b2", (C,)),
        ("y_emb", (NC, C)),
        ("ada_w", (L, 6 * C, C)), ("ada_b", (L, 6 * C)),
        ("qkv_w", (L, 3 * C, C)), ("qkv_b", (L, 3 * C)),
        ("proj_w", (L, C, C)), ("proj_b", (L, C)),
        ("fc1_w", (L, H, C)), ("fc1_b", (L, H)),
        ("fc2_w", (L, C, H)), ("fc2_b", (L, C)),
        ("f_ada_w", (2 * C, C)), ("f_ada_b", (2 * C,)),
        ("f_out_w", (P, C)), ("f_out_b", (P,)),
    ]


def num_params(cfg: DiTConfig) -> int:
    return sum(int(np.prod(s)) for _, s in param_specs(cfg))


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """(N, C, H, W) -> (N, T, C*p*p). Token = gi*gw + gj, feature = c*p*p + pi*p + pj
    (the order of a Conv2d(C, hidden, p, stride=p) weight flattened with .view(hidden, -1))."""
    N, C, H, W = x.shape
    gh, gw = H // p, W // p
    return x.reshape(N, C, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(N, gh * gw, C * p * p)


def unpatchify(x: torch.Tensor, p: int, C: int, H: int, W: int) -> torch.Tensor:
    N = x.shape[0]
    gh, gw = H // p, W // p
    return x.reshape(N, gh, gw, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """DiT's sinusoidal embedding: [cos(t f_i), sin(t f_i)], f_i = exp(-ln(max_period) i / half)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def pos_embed_2d(dim: int, gh: int, gw: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding (DiT's get_2d_sincos_pos_embed), token i = r*gw + c:
    channels [0, dim/2) encode the column, [dim/2, dim) the row, each as [sin, cos]."""
    def emb1d(d: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0))
        out = np.einsum("m,d->md", pos.reshape(-1).astype(np.float64), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    rows, cols = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    emb = np.concatenate([emb1d(dim // 2, cols), emb1d(dim // 2, rows)], axis=1)
    return torch.from_numpy(emb.astype(np.float32))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiT(nn.Module):
    """One flow-matching specialist (or the whole end-to-end model)."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        assert cfg.img_size % cfg.patch == 0 and cfg.hidden % cfg.heads == 0 and cfg.hidden % 4 == 0
        self.cfg = cfg
        for name, shape in param_specs(cfg):  # registration order == canonical order
            self.register_parameter(name, nn.Parameter(torch.empty(shape)))
        self.register_buffer("pos", pos_embed_2d(cfg.hidden, cfg.grid, cfg.grid), persistent=False)
        self.init_weights("dit")

    @torch.no_grad()
    def init_weights(self, mode: str = "dit", generator: torch.Generator | None = None) -> None:
        """"dit": adaLN-Zero initialisation. "test": every tensor random, so that every gradient
        path is non-zero (used by the C/CUDA tests)."""
        if mode == "test":
            for name, p in self.named_parameters():
                std = 0.5 if name == "y_emb" else (0.1 if name.endswith("_b") else 1.0 / math.sqrt(p.shape[-1]))
                p.copy_(torch.randn(p.shape, generator=generator) * std)
            return

        def xavier_(w: torch.Tensor) -> None:  # w: (out, in)
            bound = math.sqrt(6.0 / (w.shape[0] + w.shape[1]))
            w.uniform_(-bound, bound, generator=generator)

        for p in self.parameters():
            p.zero_()
        xavier_(self.x_emb_w)
        self.t_w1.normal_(0.0, 0.02, generator=generator)
        self.t_w2.normal_(0.0, 0.02, generator=generator)
        self.y_emb.normal_(0.0, 0.02, generator=generator)
        for l in range(self.cfg.depth):
            for w in (self.qkv_w, self.proj_w, self.fc1_w, self.fc2_w):
                xavier_(w[l])
        # adaLN and output layers stay zero (adaLN-Zero)

    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])

    def flat_grads(self) -> torch.Tensor:
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in self.parameters()])

    @torch.no_grad()
    def load_flat(self, flat: torch.Tensor) -> None:
        off = 0
        for p in self.parameters():
            p.copy_(flat[off:off + p.numel()].view_as(p))
            off += p.numel()
        assert off == flat.numel()

    def _attention(self, h: torch.Tensor, l: int) -> torch.Tensor:
        N, T, C = h.shape
        H = self.cfg.heads
        qkv = F.linear(h, self.qkv_w[l], self.qkv_b[l]).reshape(N, T, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = torch.softmax((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // H)), dim=-1)  # non-causal
        return F.linear((att @ v).transpose(1, 2).reshape(N, T, C), self.proj_w[l], self.proj_b[l])

    def forward_tokens(self, xp: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """xp (N, T, P) noisy patches, t (N,) in [0, 1], y (N,) labels -> velocity (N, T, P)."""
        cfg = self.cfg
        C = cfg.hidden
        x = F.linear(xp, self.x_emb_w, self.x_emb_b) + self.pos
        tf = timestep_embedding(t * cfg.t_scale, cfg.freq_dim)
        c = F.linear(F.silu(F.linear(tf, self.t_w1, self.t_b1)), self.t_w2, self.t_b2) + F.embedding(y, self.y_emb)
        sc = F.silu(c)
        mod = F.linear(sc, self.ada_w.reshape(-1, C), self.ada_b.reshape(-1))  # all layers in one GEMM
        for l in range(cfg.depth):
            sh1, s1, g1, sh2, s2, g2 = mod[:, l * 6 * C:(l + 1) * 6 * C].chunk(6, dim=1)
            x = x + g1.unsqueeze(1) * self._attention(modulate(F.layer_norm(x, (C,), eps=LN_EPS), sh1, s1), l)
            h = modulate(F.layer_norm(x, (C,), eps=LN_EPS), sh2, s2)
            h = F.gelu(F.linear(h, self.fc1_w[l], self.fc1_b[l]), approximate="tanh")
            x = x + g2.unsqueeze(1) * F.linear(h, self.fc2_w[l], self.fc2_b[l])
        shf, sf = F.linear(sc, self.f_ada_w, self.f_ada_b).chunk(2, dim=1)
        return F.linear(modulate(F.layer_norm(x, (C,), eps=LN_EPS), shf, sf), self.f_out_w, self.f_out_b)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Image space: x (N, C, H, W) -> velocity (N, C, H, W)."""
        cfg = self.cfg
        v = self.forward_tokens(patchify(x, cfg.patch), t, y)
        return unpatchify(v, cfg.patch, cfg.in_ch, cfg.img_size, cfg.img_size)
