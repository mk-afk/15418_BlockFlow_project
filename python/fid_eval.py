"""FID of generated samples against CIFAR-10 training statistics (clean-fid).

    python python/fid_eval.py runs/b2/samples.npy     # uint8 (N, 32, 32, 3) from csrc/sample

Prints and appends {"samples": ..., "n": ..., "fid": ...} to results/fid.jsonl.
"""
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image


def main() -> None:
    from cleanfid import fid

    path = sys.argv[1]
    images = np.load(path)
    with tempfile.TemporaryDirectory() as tmp:
        for i, im in enumerate(images):
            Image.fromarray(im).save(f"{tmp}/{i:06d}.png")
        score = fid.compute_fid(tmp, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean")
    rec = dict(samples=path, n=int(len(images)), fid=float(score))
    print(json.dumps(rec))
    os.makedirs("results", exist_ok=True)
    with open("results/fid.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
