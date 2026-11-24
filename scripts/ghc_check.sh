#!/usr/bin/env bash
# Checks and timings on one GHC machine (one 8 GB GPU), about 10 minutes:
#   1. build and record the hardware
#   2. C/CUDA vs PyTorch tests, and the parallel test with 4 ranks sharing the GPU
#   3. step times of DiT-S/2 on CIFAR-10 for B = 1, 2, 3, 4 blocks trained in turn on the GPU
#   4. the same end-to-end step in the PyTorch reference, if python3 has torch
#   ./scripts/ghc_check.sh
# Output goes to results/ghc/ (not in git): ghc_check.log, hardware.txt and one folder per run.
# Needs nvcc, mpicxx/mpirun, data/cifar10_train.bin and tests/data/*.bin (make testdata, on any machine
# with torch). Run it from local disk such as /tmp/$USER: the AFS quota is too small for the data.
# ARGS="--batch 16 --micro 8 --steps 6" shortens every training run, to try the script out.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/ghc
mkdir -p "$OUT"
export HWLOC_COMPONENTS=-gl  # stops Open MPI's "Authorization required" lines about the X display
{
for tool in nvcc mpicxx mpirun nvidia-smi python3; do
    command -v "$tool" > /dev/null || { echo "$tool is not on PATH"; exit 1; }
done
[ -r data/cifar10_train.bin ] || {
    echo "data/cifar10_train.bin is missing; make it with: python3 python/prepare_data.py cifar10 data/cifar10_train.bin"
    exit 1
}
{   echo "machine: $(hostname)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader)"
    echo "other jobs on the gpu: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
    echo "cuda: $(nvcc --version | tail -n 1)"
    echo "mpi: $(mpirun --version | head -n 1)"
} | tee "$OUT/hardware.txt"

make gpu
[ -f tests/data/tiny.bin ] && [ -f tests/data/dits.bin ] || make testdata
./build/gpu/test_dit tests/data/tiny.bin
./build/gpu/test_dit tests/data/dits.bin
mpirun --oversubscribe -np 4 ./build/gpu/test_comm

for B in 1 2 3 4; do
    ./build/gpu/train --data data/cifar10_train.bin --blocks $B --steps 30 --out "$OUT/P1_B${B}_D1" ${ARGS:-}
    rm -f "$OUT/P1_B${B}_D1"/block*.bin  # the checkpoint of a timing run is not needed (260 MB)
done
python3 python/analyze_bench.py "$OUT"/P*

if python3 -c "import torch" 2> /dev/null; then
    python3 python/train_ref.py --data data/cifar10_train.bin --steps 60 --out "$OUT/speed_pytorch" ${ARGS:-}
    rm -f "$OUT"/speed_pytorch/block*.bin
    python3 - "$OUT" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1] + "/P1_B1_D1/summary.json"))
t = json.loads(open(sys.argv[1] + "/speed_pytorch/log.jsonl").read().splitlines()[-1])["ms_per_step"]
print(f"end-to-end step, {c['batch_per_block']} images in micro-batches of {c['micro_batch']}: "
      f"C/CUDA {c['ms_max']['step']:.0f} ms, PyTorch {t:.0f} ms")
EOF
else
    echo "python3 has no torch: skipped the PyTorch speed comparison"
fi
} 2>&1 | tee "$OUT/ghc_check.log"
