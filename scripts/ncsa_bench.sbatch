#!/bin/bash
# Layout sweep on one 4-GPU node: serial blocks on 1 GPU and every P = B x D layout with P <= 4.
#   sbatch --account=<acct>-delta-gpu scripts/ncsa_bench.sbatch
# Output: results/bench_<job>/P<P>_B<B>_D<D>/summary.json, the analyze_bench.py table, hardware.txt.
# Module names differ between systems (check `module spider cuda`, `module spider openmpi`).
#SBATCH --job-name=blockflow-bench
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=01:30:00
#SBATCH --output=results/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
module load cuda openmpi 2>/dev/null || true
OUT=results/bench_${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
{ nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nvcc --version | tail -n 1; nvidia-smi topo -m; } | tee "$OUT/hardware.txt"
make gpu GPU_ARCH=sm_80 USE_NCCL=1

for layout in "1 1 1" "1 2 1" "1 3 1" "1 4 1" "2 1 2" "2 2 1" "3 3 1" "4 1 4" "4 2 2" "4 4 1"; do
    read -r P B D <<< "$layout"
    srun --ntasks="$P" --gpus-per-task=1 ./build/gpu/train --data data/cifar10_train.bin --blocks "$B" \
        --replicas "$D" --micro 128 --steps 60 --out "$OUT/P${P}_B${B}_D${D}"
done
python3 python/analyze_bench.py "$OUT"/P*
