import os
import sys
import csv
from pathlib import Path

# 把工程根目录加到 sys.path
CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.eval_metrics import eval_pair  # snr_db / pesq_wb / stoi


def r(d: dict):
    """四舍五入，美观一点，NaN 保持不变。"""
    return {k: (round(v, 4) if v == v else v) for k, v in d.items()}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="对比 Noisy / DNN / UNet 的 SNR、PESQ、STOI 指标")
    ap.add_argument("--clean", type=str, default="data/clean/example.wav")
    ap.add_argument("--noisy", type=str, default="data/noisy/example_noisy.wav")
    ap.add_argument("--dnn",   type=str, default="results/enhanced_from_ckpt.wav",
                    help="MaskNet(DNN) 增强后的 wav 路径")
    ap.add_argument("--unet",  type=str, default="results/example_unet_enh.wav",
                    help="UNet 增强后的 wav 路径")
    ap.add_argument("--out",   type=str, default="results/metrics_unet_vs_dnn.csv",
                    help="输出的 CSV 文件路径")
    ap.add_argument("--sr",    type=int, default=16000)
    args = ap.parse_args()

    clean_p = Path(args.clean)
    noisy_p = Path(args.noisy)
    dnn_p   = Path(args.dnn)
    unet_p  = Path(args.unet)

    # 基本检查
    assert clean_p.exists(), f"缺少 clean 文件: {clean_p}"
    assert noisy_p.exists(), f"缺少 noisy 文件: {noisy_p}"

    items = []

    print(">>> Evaluating ...")

    m_noisy = eval_pair(str(clean_p), str(noisy_p), sr=args.sr)
    print("[NOISY]", r(m_noisy))
    items.append(("noisy", m_noisy))

    if dnn_p.exists():
        m_dnn = eval_pair(str(clean_p), str(dnn_p), sr=args.sr)
        print("[DNN  ]", r(m_dnn))
        items.append(("dnn_mask", m_dnn))
    else:
        print(f"[Warn] DNN 结果不存在: {dnn_p}")

    if unet_p.exists():
        m_unet = eval_pair(str(clean_p), str(unet_p), sr=args.sr)
        print("[UNET ]", r(m_unet))
        items.append(("unet", m_unet))
    else:
        print(f"[Warn] UNet 结果不存在: {unet_p}")

    # 写 CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        for name, m in items:
            w.writerow([name, m.get("snr_db"), m.get("pesq_wb"), m.get("stoi")])

    print("Saved metrics to:", out_path)


if __name__ == "__main__":
    main()
