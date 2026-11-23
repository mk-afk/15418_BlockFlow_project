"""Exact parameters, all-reduce volume, training FLOPs and activation memory of DiT-S/2 layouts.

    python python/param_count.py

Counts come from the parameter layout and the activation allocator of the C/CUDA trainer
(csrc/dit.cuh: fill_param_sizes, acts_alloc). Nothing here is measured.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockflow import DiTConfig, num_params  # noqa: E402

BATCH = 512  # images per block per step


def fwd_flops_per_image(cfg: DiTConfig) -> int:
    C, T, P, F, L, H = cfg.hidden, cfg.tokens, cfg.patch_dim, cfg.freq_dim, cfg.depth, cfg.mlp_ratio * cfg.hidden
    per_layer = T * (4 * C * C + 2 * C * H) + 2 * T * T * C + 6 * C * C  # matmuls, attention, adaLN
    return 2 * (L * per_layer + 2 * T * P * C + 2 * C * C + F * C + C * C)


def activation_floats_per_image(cfg: DiTConfig) -> int:  # mirrors acts_alloc with a batch of 1
    C, T, P, F, L, NH, H = cfg.hidden, cfg.tokens, cfg.patch_dim, cfg.freq_dim, cfg.depth, cfg.heads, cfg.mlp_ratio * cfg.hidden
    per_layer = 8 * T * C + T * H + NH * T * T + 2 * T
    rest = 4 * T * P + 15 * T * C + 2 * T * H + NH * T * T + T + 1 + F + 7 * C + 12 * L * C + 4 * C + -(-T * P // 256) + 1
    return L * per_layer + rest


def main() -> None:
    full = num_params(DiTConfig())
    print(f"DiT-S/2 (CIFAR-10): {full:,} parameters; {BATCH} images per block per step")
    print(f"{'P':>2} {'B':>2} {'D':>2} {'layers':>6} {'params/block':>13} {'x full':>7} {'all-reduce MB':>14} "
          f"{'TFLOP/step':>11} {'act MB/image':>13} {'state GB':>9}")
    for P in (2, 4, 8):
        for B in (1, 2, 4):
            if B > P:
                continue
            D = P // B
            cfg = DiTConfig(depth=12 // B)
            nb = num_params(cfg)
            ring_mb = 2 * (D - 1) / D * nb * 4 / 1e6
            tflop = 3 * fwd_flops_per_image(cfg) * BATCH * B / 1e12
            act_mb = activation_floats_per_image(cfg) * 4 / 1e6
            print(f"{P:>2} {B:>2} {D:>2} {cfg.depth:>6} {nb:>13,} {nb / full:>7.3f} {ring_mb:>14.1f} {tflop:>11.1f} "
                  f"{act_mb:>13.1f} {5 * nb * 4 / 1e9:>9.2f}")
    act = activation_floats_per_image(DiTConfig()) * 4 / 1e9
    print(f"activations of the full model: {64 * act:.2f} GB for a micro-batch of 64, {128 * act:.2f} GB for 128")


if __name__ == "__main__":
    main()
