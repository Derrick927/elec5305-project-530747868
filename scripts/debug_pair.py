import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.dataset import PairDataset
from src.stft import stft
from src.masking import ideal_ratio_mask
from src.utils import load_wav


def main():
    manifest = "manifests/train.csv"
    print(f"[Info] Loading manifest: {manifest}")

    ds = PairDataset(manifest, mode="on_the_fly")

    print(f"[Info] Dataset length = {len(ds)}")

    for i in range(5):
        print("\n=======================")
        print(f"Sample #{i}")
        print("=======================")

        noisy_T, irm_T = ds[i]
        print(f"noisy_T shape = {noisy_T.shape}")
        print(f"irm_T   shape = {irm_T.shape}")

        print(f"noisy_T min/max = {noisy_T.min():.4f} / {noisy_T.max():.4f}")
        print(f"irm_T   min/max = {irm_T.min():.4f} / {irm_T.max():.4f}")

        if irm_T.min() < 0 or irm_T.max() > 1:
            print("[Warning] IRM mask not in [0,1] range - potential issue!")

        if noisy_T.shape != irm_T.shape:
            print("[Error] Feature and label shapes do not match!")

    print("\n[Done] Debug complete, check output for issues.")


if __name__ == "__main__":
    main()
