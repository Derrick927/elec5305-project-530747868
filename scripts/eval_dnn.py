# scripts/eval_dnn.py
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.eval_metrics import eval_pair

SR = 16000

CLEAN   = "data/clean/example.wav"
NOISY   = "data/noisy/example_noisy.wav"
DNN_OUT = "results/example_dnn.wav"

def r(d):  # keep NaN 
    return {k: (round(v, 4) if v == v else v) for k, v in d.items()}

def main():
    # basic check
    assert Path(CLEAN).exists(),   "Missing clean audio"
    assert Path(NOISY).exists(),   "Missing noisy audio"
    assert Path(DNN_OUT).exists(), "Missing DNN output, run train_mask.py first"

    print(">>> Evaluating DNN")
    m_noisy = eval_pair(CLEAN, NOISY, SR)
    m_dnn   = eval_pair(CLEAN, DNN_OUT, SR)

    print("[NOISY] ", r(m_noisy))
    print("[DNN]   ", r(m_dnn))

    # write outcome in CSV
    out = Path("results/metrics.csv")
    out.parent.mkdir(exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["dnn", m_dnn["snr_db"], m_dnn["pesq_wb"], m_dnn["stoi"]])
    print("Appended DNN metrics to:", out)

if __name__ == "__main__":
    main()
