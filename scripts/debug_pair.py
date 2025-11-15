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

    # 取前 5 条，逐条检查
    for i in range(5):
        print("\n=======================")
        print(f"Sample #{i}")
        print("=======================")

        noisy_T, irm_T = ds[i]  # noisy_mag, irm_mag  (T, F)
        print(f"noisy_T shape = {noisy_T.shape}")
        print(f"irm_T   shape = {irm_T.shape}")

        # 基本数值检查
        print(f"noisy_T min/max = {noisy_T.min():.4f} / {noisy_T.max():.4f}")
        print(f"irm_T   min/max = {irm_T.min():.4f} / {irm_T.max():.4f}")

        # 检查 IRM 是否在 0~1
        if irm_T.min() < 0 or irm_T.max() > 1:
            print("⚠️  IRM 掩蔽不在 [0,1] 范围内 —— 有重大问题！")

        # 检查长度
        if noisy_T.shape != irm_T.shape:
            print("⚠️  特征和标签 shape 不一致 —— 非常错误！")

    print("\n[Done] Debug 完成，请查看输出判断问题。")


if __name__ == "__main__":
    main()
