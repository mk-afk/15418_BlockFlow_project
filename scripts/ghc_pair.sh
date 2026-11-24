#!/usr/bin/env bash
# First multi-GPU measurements: two GHC machines, one GPU each, MPI over campus Ethernet (about 5 minutes).
# Run after ghc_check.sh on one machine, naming a second one of the same kind:
#   ./scripts/ghc_pair.sh ghc42
#   1. the parallel test with one rank on each machine
#   2. B = 2 with one block on each machine (the blocks never communicate)
#   3. end-to-end training with one replica on each machine (gradient all-reduce over Ethernet)
# and the table of all runs in results/ghc, with speedups against the one-GPU runs of ghc_check.sh.
# Output goes to results/ghc/ (not in git): ghc_pair.log, hardware_pair.txt and one folder per run.
# Needs ssh from this machine to the other without a password prompt, and this directory with its build/
# and data/ at the same path on both: a shared folder, or a copy on each machine's local disk.
set -euo pipefail
cd "$(dirname "$0")/.."
OTHER=${1:?usage: ghc_pair.sh OTHER_MACHINE}
OUT=results/ghc
mkdir -p "$OUT"
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export HWLOC_COMPONENTS=-gl  # stops Open MPI's "Authorization required" lines about the X display
RUN="$(command -v mpirun) -np 2 --host $(hostname),$OTHER -x PATH -x LD_LIBRARY_PATH -x HWLOC_COMPONENTS"
{
if ! $RUN sh -c 'test -x build/gpu/train && test -r data/cifar10_train.bin && touch results/ghc/.$$ && rm results/ghc/.$$ &&
        echo "machine: $(hostname), gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"' |
        tee "$OUT/hardware_pair.txt"; then
    echo "Could not start on both machines, or $OTHER cannot read build/gpu and data/ or write to results/ghc."
    echo "Check that 'ssh $OTHER true' works without a password prompt and that this directory exists there too."
    exit 1
fi
dev=$(ip -o route get "$(getent hosts "$OTHER" | awk '{print $1; exit}')" 2> /dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p') || true
echo "link to $OTHER: ${dev:-?}, $(cat "/sys/class/net/${dev:-none}/speed" 2> /dev/null || echo '?') Mb/s" | tee -a "$OUT/hardware_pair.txt"

$RUN ./build/gpu/test_comm
$RUN ./build/gpu/train --data data/cifar10_train.bin --blocks 2 --steps 30 --out "$OUT/P2_B2_D1" ${ARGS:-}
$RUN ./build/gpu/train --data data/cifar10_train.bin --blocks 1 --steps 30 --out "$OUT/P2_B1_D2" ${ARGS:-}
rm -f "$OUT"/P2_*/block*.bin
python3 python/analyze_bench.py "$OUT"/P*
} 2>&1 | tee "$OUT/ghc_pair.log"
