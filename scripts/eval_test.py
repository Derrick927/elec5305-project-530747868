# scripts/eval_test.py
import sys, os, csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.eval_metrics import eval_pair

CLEAN = "data/clean/example.wav"
NOISY = "data/noisy/example_noisy.wav"
DENOISED = "results/example_denoised.wav"

def main():
    # Path existence check
    for p in [CLEAN, NOISY, DENOISED]:
        for p in [CLEAN, NOISY, DENOISED]:
        assert Path(p).exists(), f"Missing file: {p}\nRun: python3 scripts/noise_test.py to generate noisy and denoised results"


    print(">>> Evaluating...")
    m_noisy = eval_pair(CLEAN, NOISY)
    m_deno = eval_pair(CLEAN, DENOISED)

    # Terminal printing
    print("\n[NOISY]")
    print({k: round(v, 4) if v == v else v for k, v in m_noisy.items()})  # v==v filter NaN
    print("\n[DENOISED]")
    print({k: round(v, 4) if v == v else v for k, v in m_deno.items()})

    # output CSV
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "snr_db", "pesq_wb", "stoi"])
        w.writerow(["noisy", m_noisy["snr_db"], m_noisy["pesq_wb"], m_noisy["stoi"]])
        w.writerow(["denoised", m_deno["snr_db"], m_deno["pesq_wb"], m_deno["stoi"]])
    print("\nSaved metrics to:", csv_path)

if __name__ == "__main__":
    main()
