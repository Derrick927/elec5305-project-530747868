# scripts/eval_wiener.py
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.eval_metrics import eval_pair

SR = 16000

CLEAN     = "data/clean/example.wav"
NOISY     = "data/noisy/example_noisy.wav"
SUBTRACT  = "results/example_denoised.wav"   # Spectral subtraction result (optional)
WIENER    = "results/example_wiener.wav"     # Wiener result (optional)

def r(d):  # round helper
    return {k: (round(v, 4) if v == v else v) for k, v in d.items()}  # v==v 过滤 NaN

def main():
    # --- check inputs ---
    for p in [CLEAN, NOISY]:
        assert Path(p).exists(), f"Missing file: {p}"
    have_sub    = Path(SUBTRACT).exists()
    have_wiener = Path(WIENER).exists()

    print(">>> Evaluating ...")
    m_noisy  = eval_pair(CLEAN, NOISY, SR)
    m_wiener = eval_pair(CLEAN, WIENER, SR)   if have_wiener else None
    m_sub    = eval_pair(CLEAN, SUBTRACT, SR) if have_sub    else None

    print("[NOISY]   ", r(m_noisy))
    if m_wiener: print("[WIENER]  ", r(m_wiener))
    if m_sub:    print("[SUBTRACT]", r(m_sub))

    # --- write CSV ---
    Path("results").mkdir(exist_ok=True)
    out = Path("results/metrics.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy",   m_noisy["snr_db"],  m_noisy["pesq_wb"],  m_noisy["stoi"]])
        if m_wiener:
            w.writerow(["wiener",  m_wiener["snr_db"], m_wiener["pesq_wb"], m_wiener["stoi"]])
        if m_sub:
            w.writerow(["subtract", m_sub["snr_db"],  m_sub["pesq_wb"],  m_sub["stoi"]])
    print("Saved metrics to:", out)

if __name__ == "__main__":
    main()
