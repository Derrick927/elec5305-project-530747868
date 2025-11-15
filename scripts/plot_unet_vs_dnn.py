import os
import sys
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
画两条训练曲线：
- DNN: train_log.csv
- UNet: train_log_unet.csv

输出: results/train_curve_unet_vs_dnn.png
"""

def load_curve(path: Path, key_train: str, key_val: str):
    assert path.exists(), f"CSV 文件不存在: {path}"
    epochs, tr, va = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr.append(float(row[key_train]))
            v = row.get(key_val, "")
            if v == "" or v is None:
                va.append(float("nan"))
            else:
                va.append(float(v))
    return epochs, tr, va


def main():
    import argparse
    ap = argparse.ArgumentParser(description="绘制 UNet vs DNN 训练曲线")
    ap.add_argument("--dnn-log", type=str, default="checkpoints/demo/train_log.csv",
                    help="DNN 训练日志（train_mask 的日志）")
    ap.add_argument("--unet-log", type=str, default="checkpoints/unet/train_log_unet.csv",
                    help="UNet 训练日志")
    ap.add_argument("--out", type=str, default="results/train_curve_unet_vs_dnn.png")
    args = ap.parse_args()

    dnn_path = Path(args.dnn_log)
    unet_path = Path(args.unet_log)

    # ---------- 读取 DNN ----------
    dnn_epochs, dnn_tr, dnn_va = load_curve(
        dnn_path, key_train="train_bce", key_val="val_bce"
    )

    # ---------- 读取 UNet ----------
    unet_epochs, unet_tr, unet_va = load_curve(
        unet_path, key_train="train_loss", key_val="val_loss"
    )

    # ---------- 画图 ----------
    plt.figure(figsize=(10,5))

    # --- DNN ---
    plt.plot(dnn_epochs, dnn_tr, label="DNN Train BCE", linestyle="-")
    plt.plot(dnn_epochs, dnn_va, label="DNN Val BCE", linestyle="--")

    # --- UNet ---
    plt.plot(unet_epochs, unet_tr, label="UNet Train MSE", linestyle="-")
    plt.plot(unet_epochs, unet_va, label="UNet Val MSE", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves: UNet vs DNN")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print("Saved plot to:", out_path)


if __name__ == "__main__":
    main()
