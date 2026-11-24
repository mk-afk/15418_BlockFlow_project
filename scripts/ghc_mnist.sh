#!/usr/bin/env bash
# MNIST go/no-go runs on a GHC GPU: DiT-S/2 trained end-to-end and as B = 2 blocks by the C/CUDA trainer,
# and end-to-end by the PyTorch reference, then an 8x8 grid of samples from each in results/figures/.
#   ./scripts/ghc_mnist.sh           the three runs in turn, then the grids
#   ./scripts/ghc_mnist.sh c_b2      one run (c_e2e, c_b2 or pytorch), to use three machines at once
#   ./scripts/ghc_mnist.sh grids     the grids, once the three runs are done
# 3000 steps of batch 128 in micro-batches of 32, which fits both trainers on 8 GB; ARGS="--steps 1000"
# shortens the runs. Needs data/mnist_train.bin and python3 with torch. Each run leaves 260 MB of
# checkpoints in runs/ (not in git).
set -euo pipefail
cd "$(dirname "$0")/.."
export HWLOC_COMPONENTS=-gl  # stops Open MPI's "Authorization required" lines about the X display
COMMON="--data data/mnist_train.bin --batch 128 --micro 32 --steps 3000 ${ARGS:-}"
make gpu
for run in ${@:-c_e2e c_b2 pytorch grids}; do
    case $run in
        c_e2e) ./build/gpu/train $COMMON --out runs/mnist_c_e2e ;;
        c_b2) ./build/gpu/train $COMMON --blocks 2 --out runs/mnist_c_b2 ;;
        pytorch) python3 python/train_ref.py $COMMON --out runs/mnist_pytorch ;;
        grids)
            mkdir -p results/figures
            for r in pytorch c_e2e c_b2; do
                python3 python/sample_ref.py --ckpt runs/mnist_$r --n 64 --grid results/figures/mnist_$r.png
            done ;;
        *) echo "unknown run '$run' (c_e2e, c_b2, pytorch or grids)"; exit 1 ;;
    esac
done
