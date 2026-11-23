#!/usr/bin/env bash
# Checks and timings on one GHC machine (one 8 GB GPU), about 20 minutes:
#   1. build with the MPI all-reduce and record the hardware
#   2. C/CUDA vs PyTorch tests, and the parallel test with 4 ranks sharing the GPU
#   3. step times of DiT-S/2 on CIFAR-10 for B = 1, 2, 3, 4 blocks trained in turn on the GPU
#   4. the same end-to-end step in C/CUDA and in the PyTorch reference (micro-batch 32 fits both)
#   ./scripts/ghc_check.sh
# Output goes to results/ghc/ (not in git): ghc_check.log, hardware.txt and one folder per run.
# Needs nvcc, mpicxx/mpirun, python3 with torch + numpy, and data/cifar10_train.bin.
# ARGS="--batch 16 --micro 8 --steps 6" shortens every training run, to try the script out.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/ghc
mkdir -p "$OUT"
{
for tool in nvcc mpicxx mpirun nvidia-smi python3; do
    command -v "$tool" > /dev/null || { echo "$tool is not on PATH"; exit 1; }
done
[ -r data/cifar10_train.bin ] || {
    echo "data/cifar10_train.bin is missing; make it with: python3 python/prepare_data.py cifar10 data/cifar10_train.bin"
    exit 1
}
{   echo "machine: $(hostname)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
    echo "other jobs on the gpu: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
    echo "cuda: $(nvcc --version | tail -n 1)"
    echo "mpi: $(mpirun --version | head -n 1)"
} | tee "$OUT/hardware.txt"

make gpu
make testdata
./build/gpu/test_dit tests/data/tiny.bin
./build/gpu/test_dit tests/data/dits.bin
mpirun --oversubscribe -np 4 ./build/gpu/test_comm

for B in 1 2 3 4; do
    ./build/gpu/train --data data/cifar10_train.bin --blocks $B --steps 30 --out "$OUT/P1_B${B}_D1" ${ARGS:-}
done
./build/gpu/train --data data/cifar10_train.bin --micro 32 --steps 30 --out "$OUT/speed_c" ${ARGS:-}
python3 python/train_ref.py --data data/cifar10_train.bin --micro 32 --steps 60 --out "$OUT/speed_pytorch" ${ARGS:-}
rm -f "$OUT"/*/block*.bin  # the checkpoints of timing runs are not needed (260 MB each)

python3 python/analyze_bench.py "$OUT"/P*
python3 - "$OUT" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/speed_c/summary.json"))
t = json.loads(open(sys.argv[1] + "/speed_pytorch/log.jsonl").read().splitlines()[-1])["ms_per_step"]
print(f"end-to-end step, {c['batch_per_block']} images in micro-batches of {c['micro_batch']}: "
      f"C/CUDA {c['ms_max']['step']:.0f} ms, PyTorch {t:.0f} ms")
EOF
} 2>&1 | tee "$OUT/ghc_check.log"
