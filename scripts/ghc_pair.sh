#!/usr/bin/env bash
# First multi-GPU measurements: two GHC machines, one GPU each, MPI over campus Ethernet (about 5 minutes).
# Run after ghc_check.sh on one machine, naming a second one of the same kind:
#   ./scripts/ghc_pair.sh ghc42
#   1. the parallel test with one rank on each machine
#   2. B = 2 with one block on each machine (the blocks never communicate)
#   3. end-to-end training with one replica on each machine (gradient all-reduce over Ethernet)
# and the table of all runs in results/ghc, with speedups against the one-GPU runs of ghc_check.sh.
# Output goes to results/ghc/ (not in git): ghc_pair.log, hardware_pair.txt and one folder per run.
# Needs this directory with its build/ and data/ at the same path on both machines (a copy on each local
# disk), and ssh to the other machine without a password prompt. GHC always asks for one, so log in once
# over a shared connection and tell mpirun to use it:
#   ssh -M -S /tmp/$USER/cm -fN ghc42 && export OMPI_MCA_plm_rsh_args="-S /tmp/$USER/cm"
#   rsync -a --exclude runs -e "ssh -S /tmp/$USER/cm" ./ ghc42:$PWD/
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
        echo "machine: $(hostname), gpu: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
            ), other jobs on the gpu: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"' |
        tee "$OUT/hardware_pair.txt"; then
    echo "Could not start on both machines, or $OTHER cannot read build/gpu and data/ or write to results/ghc."
    echo "Check that ssh to $OTHER works without a password prompt (see the top of this script) and that this"
    echo "directory exists there too."
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
