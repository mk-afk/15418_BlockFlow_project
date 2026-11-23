"""Dataset files read by both trainers:

    int32 header[16]   magic, version, N, C, H, W, num_classes, 0...
    uint8 labels[N]
    uint8 images[N*C*H*W]   (NCHW)
"""
from __future__ import annotations

import numpy as np

DATA_MAGIC = 4180001


def write_dataset(path: str, images: np.ndarray, labels: np.ndarray, num_classes: int) -> None:
    hdr = np.zeros(16, dtype=np.int32)
    hdr[:7] = [DATA_MAGIC, 1, *images.shape, num_classes]
    with open(path, "wb") as f:
        f.write(hdr.tobytes())
        f.write(np.ascontiguousarray(labels, dtype=np.uint8).tobytes())
        f.write(np.ascontiguousarray(images, dtype=np.uint8).tobytes())


def read_dataset(path: str):
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(64), dtype=np.int32)
        assert hdr[0] == DATA_MAGIC, f"{path}: not a BlockFlow dataset file"
        N, C, H, W, nc = (int(v) for v in hdr[2:7])
        labels = np.frombuffer(f.read(N), dtype=np.uint8).copy()
        images = np.frombuffer(f.read(N * C * H * W), dtype=np.uint8).reshape(N, C, H, W).copy()
    return images, labels, nc


def prepare(name: str, out: str, split: str = "train", root: str = "data/raw") -> None:
    """Downloads MNIST (zero-padded to 32x32) or CIFAR-10 with torchvision and writes a dataset file."""
    import torchvision

    train = split == "train"
    if name == "mnist":
        ds = torchvision.datasets.MNIST(root, train=train, download=True)
        images = np.pad(ds.data.numpy()[:, None], ((0, 0), (0, 0), (2, 2), (2, 2)))
        labels = ds.targets.numpy()
    else:
        ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
        images, labels = ds.data.transpose(0, 3, 1, 2), np.asarray(ds.targets)
    write_dataset(out, images, labels, 10)
    print(f"wrote {out}: {images.shape[0]} images of shape {images.shape[1:]}")
