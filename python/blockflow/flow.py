"""Flow matching and the block-dispatched Euler sampler.

x_t = (1 - t) x_0 + t eps, target velocity v = eps - x_0.  Sampling integrates dx/dt = v from
t = 1 (noise) to t = 0 (data), and each Euler step runs only the block whose interval contains t.
"""
from __future__ import annotations

import torch

from .partition import TimePartition


def flow_targets(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - tt) * x0 + tt * noise, noise - x0


def euler_times(steps: int) -> list[float]:
    """t_k = 1 - k/steps in double (the C sampler uses the same formula)."""
    return [1.0 - k / steps for k in range(steps + 1)]


@torch.no_grad()
def euler_sample(models, partition: TimePartition, x1: torch.Tensor, y: torch.Tensor, steps: int,
                 block_trace: list | None = None) -> torch.Tensor:
    """models[b] is the specialist for block b."""
    x = x1.clone()
    ts = euler_times(steps)
    for k in range(steps):
        t, t_next = ts[k], ts[k + 1]
        b = partition.block_of(t)
        if block_trace is not None:
            block_trace.append(b)
        v = models[b](x, torch.full((x.shape[0],), t, dtype=torch.float32, device=x.device), y)
        x = x + (t_next - t) * v
    return x
