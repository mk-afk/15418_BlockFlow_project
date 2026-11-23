"""Equal-probability partition of the training-time distribution into B blocks.

Training times follow a logit-normal distribution, t = sigmoid(m + s*z),
z ~ N(0, 1) (Esser et al., 2024).  Block b owns the interval between the
b/B and (b+1)/B quantiles, so every specialist receives the same share of the
end-to-end training distribution.  For B = 4, m = 0, s = 1 the inner
boundaries are t ~= 0.3375, 0.5, 0.6625.

The inverse normal CDF is Wichura's AS241 (the algorithm used by Python's
statistics.NormalDist.inv_cdf); csrc/common.h implements the same algorithm
so C and Python agree on the block boundaries to ~1e-16.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import torch

_STD_NORMAL = NormalDist()


def logit_normal_ppf(q: float, mean: float = 0.0, std: float = 1.0) -> float:
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    z = _STD_NORMAL.inv_cdf(q)
    return 1.0 / (1.0 + math.exp(-(mean + std * z)))


class TimePartition:
    def __init__(self, num_blocks: int, mean: float = 0.0, std: float = 1.0):
        assert num_blocks >= 1
        self.B, self.mean, self.std = num_blocks, mean, std
        self.quantiles = [b / num_blocks for b in range(num_blocks + 1)]
        self.edges = [logit_normal_ppf(q, mean, std) for q in self.quantiles]

    def interval(self, b: int) -> tuple[float, float]:
        return self.edges[b], self.edges[b + 1]

    def quantile_interval(self, b: int) -> tuple[float, float]:
        return self.quantiles[b], self.quantiles[b + 1]

    def block_of(self, t: float) -> int:
        """Block whose interval [edge_b, edge_{b+1}) contains t (t = 1 belongs to the last block)."""
        for b in range(self.B - 1, 0, -1):
            if t >= self.edges[b]:
                return b
        return 0

    def sample_t(self, b: int, n: int, generator: torch.Generator | None = None,
                 device: torch.device | str | None = None) -> torch.Tensor:
        """n training times for block b: inverse-CDF sampling restricted to the block's quantile range."""
        q_lo, q_hi = self.quantile_interval(b)
        u = q_lo + (q_hi - q_lo) * torch.rand(n, generator=generator, dtype=torch.float64)
        u = u.clamp(1e-12, 1.0 - 1e-12)
        t = torch.sigmoid(self.mean + self.std * torch.special.ndtri(u))
        return t.to(torch.float32).to(device) if device is not None else t.to(torch.float32)

    def __repr__(self) -> str:
        e = ", ".join(f"{x:.4f}" for x in self.edges)
        return f"TimePartition(B={self.B}, mean={self.mean}, std={self.std}, edges=[{e}])"
