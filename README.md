# BlockFlow

15-418/618 F26 project, Ritvik Sood and Michael Kireeff.

Trains a DiT-S/2 flow-matching model in C/CUDA. The 12 layers can be split into B blocks that
train independently, each on its own range of timesteps, with D data-parallel replicas per block.

## Build

    make gpu         # needs nvcc and mpicxx
    make test-gpu    # needs python3 with torch to make the test data
    make test        # same tests on the CPU, no GPU needed

The defaults are for the GHC machines: sm_75, MPI all-reduce, micro-batch 64 (fits in 8 GB).
On NCSA use `make gpu GPU_ARCH=sm_80 USE_NCCL=1`.

## Run

    python3 python/prepare_data.py cifar10 data/cifar10_train.bin
    ./build/gpu/train --data data/cifar10_train.bin --blocks 2 --out runs/b2
    ./build/gpu/sample --ckpt runs/b2 --n 10000 --out runs/b2/samples.npy
    python3 python/fid_eval.py runs/b2/samples.npy

Multi-GPU is one MPI rank per GPU:

    mpirun -np 2 --host ghc41,ghc42 ./build/gpu/train --data data/cifar10_train.bin --blocks 2   # two machines
    mpirun -np 4 ./build/gpu/train --data data/cifar10_train.bin --blocks 2 --replicas 2         # one 4-GPU node

Options: `--blocks`, `--replicas`, `--batch` (per block, 512), `--micro` (per GPU, 64), `--steps`, `--out`.
Everything else is a constant at the top of `csrc/trainer.cuh`.

## Scripts

    scripts/ghc_check.sh         tests and step times on one GHC machine
    scripts/ghc_mnist.sh         MNIST runs and sample grids
    scripts/ghc_pair.sh ghc42    the same on two GHC machines
    scripts/ncsa_*.sbatch        NCSA jobs: layout sweep, CIFAR-10 + FID

## Files

    csrc/      kernels.cuh, dit.cuh (model), comm.h (MPI/NCCL), trainer.cuh, train.cu, sample.cu,
               test_dit.cu, test_comm.cu, cuda_emu.h (runs the kernels on the CPU)
    python/    PyTorch reference in blockflow/, plus test data, reference trainer and sampler, FID, data prep
