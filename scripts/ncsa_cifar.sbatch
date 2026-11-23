#!/bin/bash
# CIFAR-10 training for FID, one layout per job, then FID-10k of the final and every snapshot checkpoint.
#   sbatch --account=<acct>-delta-gpu --export=ALL,B=1,D=4 scripts/ncsa_cifar.sbatch
#   sbatch --account=<acct>-delta-gpu --export=ALL,B=2,D=2 scripts/ncsa_cifar.sbatch
#SBATCH --job-name=blockflow-cifar
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=08:00:00
#SBATCH --output=results/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
module load cuda openmpi 2>/dev/null || true
B=${B:-1}
D=${D:-4}
STEPS=${STEPS:-40000}
RUN=runs/cifar_B${B}_D${D}
make gpu GPU_ARCH=sm_80 USE_NCCL=1

srun --ntasks=$((B * D)) --gpus-per-task=1 ./build/gpu/train --data data/cifar10_train.bin --blocks "$B" \
    --replicas "$D" --micro 128 --steps "$STEPS" --out "$RUN"
for ckpt in "$RUN"/step*/ "$RUN"/; do
    ./build/gpu/sample --ckpt "$ckpt" --n 10000 --out "$ckpt/samples.npy"
    python3 python/fid_eval.py "$ckpt/samples.npy"
done
