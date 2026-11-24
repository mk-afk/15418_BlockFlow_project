# BlockFlow

15-418/618 F26 project, Ritvik Sood and Michael Kireeff.

Trains a DiT-S/2 flow-matching model in C/CUDA. The 12 layers can be split into B blocks that
train independently, each on its own range of timesteps, with D data-parallel replicas per block.

## Build

    make gpu         # needs nvcc and mpicxx
    make test-gpu    # needs python3 with torch to make the test data
    make test        # same tests on the CPU, no GPU needed

The defaults are for the GHC machines (RTX 2080, CUDA 11.7): sm_75, MPI all-reduce, micro-batch 32,
and g++-11 as nvcc's host compiler because CUDA 11 cannot use GCC 13. On GHC work in `/tmp/$USER`;
the AFS quota is too small for the data and checkpoints. `make testdata` needs torch, so run it on
another machine and copy `tests/data/` over if GHC has none.
On NCSA use `make gpu GPU_ARCH=sm_80 USE_NCCL=1` and `--micro 128`.

## Run

    python3 python/prepare_data.py cifar10 data/cifar10_train.bin
    ./build/gpu/train --data data/cifar10_train.bin --blocks 2 --out runs/b2
    ./build/gpu/sample --ckpt runs/b2 --n 10000 --out runs/b2/samples.npy
    python3 python/fid_eval.py runs/b2/samples.npy

Multi-GPU is one MPI rank per GPU:

    mpirun -np 2 --host ghc41,ghc42 ./build/gpu/train --data data/cifar10_train.bin --blocks 2   # two machines
    mpirun -np 4 ./build/gpu/train --data data/cifar10_train.bin --blocks 2 --replicas 2         # one 4-GPU node

GHC asks for a password on every ssh, which mpirun can't answer. Log in once over a shared connection
and point mpirun at it:

    ssh -M -S /tmp/$USER/cm -fN ghc42
    export OMPI_MCA_plm_rsh_args="-S /tmp/$USER/cm"

Options: `--blocks`, `--replicas`, `--batch` (per block, 512), `--micro` (per GPU, 32), `--steps`, `--out`.
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
