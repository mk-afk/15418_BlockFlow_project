"""Table of benchmark runs of csrc/train (each run directory holds summary.json).

    python python/analyze_bench.py results/ghc/P*

P is the number of ranks. Times are ms per step on the slowest rank. Speedups are against the same B
trained serially on one GPU (block-only speedup) and against end-to-end training on one GPU. A run
whose ranks shared a GPU is marked with * and is not a multi-GPU timing.
"""
import json
import os
import sys


def main() -> None:
    rows = []
    for d in sys.argv[1:]:
        f = os.path.join(d, "summary.json")
        if os.path.exists(f):
            rows.append(dict(json.load(open(f)), run=os.path.basename(os.path.normpath(d))))
    rows.sort(key=lambda r: (r["world_size"], r["blocks"], r["replicas"]))
    serial = {r["blocks"]: r["ms_max"]["step"] for r in rows if r["world_size"] == 1}
    print(f"{'P':>3} {'B':>2} {'D':>2} {'step':>9} {'compute':>9} {'allred':>8} {'optim':>7} {'img/s':>8} "
          f"{'vs serial':>9} {'vs 1-GPU':>8}  machines, device")
    for r in rows:
        m = r["ms_max"]
        vs_serial = serial[r["blocks"]] / m["step"] if r["blocks"] in serial else float("nan")
        vs_e2e = serial[1] / m["step"] if 1 in serial else float("nan")
        shared = "*" if r.get("gpus", r["world_size"]) < r["world_size"] else " "
        print(f"{r['world_size']:>2}{shared} {r['blocks']:>2} {r['replicas']:>2} {m['step']:>9.1f} {m['compute']:>9.1f} "
              f"{m['allreduce']:>8.1f} {m['optimizer']:>7.1f} {r['images_per_second']:>8.0f} {vs_serial:>9.2f} "
              f"{vs_e2e:>8.2f}  {r.get('hosts', 1)}, {r['device']}")


if __name__ == "__main__":
    main()
