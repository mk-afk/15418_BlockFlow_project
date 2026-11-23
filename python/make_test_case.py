"""Exports a PyTorch reference test case for csrc/test_dit.cu.

    python python/make_test_case.py --config tiny --out tests/data/tiny.bin

The file holds a randomly initialised model, one batch, and PyTorch's results for the training-batch
construction, the forward pass, the loss, every parameter gradient, K AdamW steps (loss per step), and
block-dispatched Euler sampling with two blocks.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockflow import DiT, DiTConfig, TimePartition, euler_sample, flow_targets, num_params, patchify  # noqa: E402

CONFIGS = {  # model, batch, AdamW steps, sampler (blocks, images, Euler steps)
    "tiny": (dict(img_size=8, in_ch=3, hidden=48, depth=2, heads=3, freq_dim=32), 4, 10, (2, 3, 8)),
    "dits": (dict(depth=1), 2, 2, (2, 2, 4)),  # DiT-S/2 layer shapes: 256 tokens, hidden 384, 6 heads
}
ADAM = (1e-3, 0.9, 0.999, 1e-8, 0.01)  # lr, beta1, beta2, eps, weight decay used by the test


def f32(x) -> bytes:
    return np.ascontiguousarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype=np.float32).tobytes()


def i32(x) -> bytes:
    return np.ascontiguousarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype=np.int32).tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny", choices=CONFIGS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mcfg, N, K, (Bs, Ns, Ks) = CONFIGS[args.config]
    cfg = DiTConfig(**mcfg)
    g = torch.Generator().manual_seed(1234)
    S, Cin, p = cfg.img_size, cfg.in_ch, cfg.patch

    model = DiT(cfg)
    model.init_weights("test", generator=g)
    params0 = model.flat_params().clone()

    # training batch: images[idx] -> x0 in [-1, 1], x_t and v, in patch layout
    images = torch.randint(0, 256, (N, Cin, S, S), dtype=torch.uint8, generator=g)
    idx = torch.randperm(N, generator=g)
    t = torch.rand(N, generator=g) * 0.98 + 0.01
    y = torch.randint(0, cfg.num_classes, (N,), generator=g)
    noise = torch.randn(N, Cin, S, S, generator=g)
    xt, v = flow_targets(images[idx].float() / 127.5 - 1.0, noise, t)
    xp, target = patchify(xt, p), patchify(v, p)

    out = model.forward_tokens(xp, t, y)
    loss = F.mse_loss(out, target)
    loss.backward()
    grads = model.flat_grads()

    m2 = DiT(cfg)
    m2.load_flat(params0)
    lr, b1, b2, eps, wd = ADAM
    opt = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd, foreach=False)
    losses = []
    for _ in range(K):
        opt.zero_grad()
        lk = F.mse_loss(m2.forward_tokens(xp, t, y), target)
        lk.backward()
        losses.append(lk.item())
        opt.step()

    blocks = []
    for _ in range(Bs):
        mb = DiT(cfg)
        mb.init_weights("test", generator=g)
        blocks.append(mb)
    x1 = torch.randn(Ns, Cin, S, S, generator=g)
    ys = torch.randint(0, cfg.num_classes, (Ns,), generator=g)
    trace = []
    xs = euler_sample(blocks, TimePartition(Bs), x1, ys, Ks, block_trace=trace)

    hdr = np.zeros(32, dtype=np.int32)
    hdr[:17] = [4180777, 2, S, Cin, p, cfg.hidden, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.num_classes,
                cfg.freq_dim, N, num_params(cfg), K, Bs, Ns, Ks]
    fh = np.zeros(16, dtype=np.float32)
    fh[:6] = [cfg.t_scale, *ADAM]
    img_bytes = images.numpy().tobytes()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(hdr.tobytes())
        f.write(fh.tobytes())
        f.write(f32(params0))
        f.write(img_bytes + b"\0" * ((-len(img_bytes)) % 4))
        for part in (i32(idx), i32(y), f32(t), f32(noise), f32(xp), f32(target), f32(out.detach()),
                     f32(np.array([loss.item()])), f32(grads), f32(np.array(losses))):
            f.write(part)
        for mb in blocks:
            f.write(f32(mb.flat_params()))
        f.write(f32(x1) + i32(ys) + f32(xs) + i32(np.array(trace)))
    print(f"wrote {args.out}: {args.config}, {num_params(cfg)} params, loss {loss.item():.6f}, blocks per step {trace}")


if __name__ == "__main__":
    main()
