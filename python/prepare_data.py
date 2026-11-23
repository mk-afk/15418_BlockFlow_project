"""Downloads MNIST or CIFAR-10 with torchvision and writes a BlockFlow dataset file.

    python python/prepare_data.py cifar10 data/cifar10_train.bin
    python python/prepare_data.py mnist data/mnist_train.bin          # padded to 32x32
    python python/prepare_data.py cifar10 data/cifar10_test.bin test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockflow.data import prepare  # noqa: E402

if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    prepare(name, out, split=sys.argv[3] if len(sys.argv) > 3 else "train")
