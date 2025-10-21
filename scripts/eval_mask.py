# scripts/eval_mask.py
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.eval_metrics import eval_pair

SR = 16000

CLEAN    = "data/clean/example.wav"
NOISY    = "data/noisy/example_noisy.wav"
SUBTRACT = "results/example_denoised.wav"   # Spectral subtraction
WIENER   = "results/example_wiener.wav"     # Wiener
MASK_IRM = "results/example_mask_irm.wav"   # IRM masking

def r(d):  # round helper; keep NaN unchanged
    return {k: (round(v, 4) if v == v else v) for k, v in d.items()}

def main():
    # basic checks
    assert Path(CLEAN).exists(), "Missing clean audio, please check."
    assert Path(NOISY).exists(), "Missing noisy audio, run noise_test.py first."

    # collect available items
    items = []
    items.append(("noisy",    eval_pair(CLEAN, NOISY)))
    if Path(WIENER).exists():
        items.append(("wiener",   eval_pair(CLEAN, WIENER)))
    if Path(SUBTRACT).exists():
        items.append(("subtract", eval_pair(CLEAN, SUBTRACT)))
    if Path(MASK_IRM).exists():
        items.append(("mask_irm", eval_pair(CLEAN, MASK_IRM)))

    print(">>> Evaluating ...")
    for name, metrics in items:
        print(f"[{name.upper():9s}]", r(metrics))

    # write CSV
    out = Path("results/metrics.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        for name, m in items:
            w.writerow([name, m["snr_db"], m["pesq_wb"], m["stoi"]])
    print("Saved metrics to:", out)

if __name__ == "__main__":
    main()
