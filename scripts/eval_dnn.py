# scripts/eval_dnn.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from src.eval_metrics import eval_pair

CLEAN   = "data/clean/example.wav"
NOISY   = "data/noisy/example_noisy.wav"
DNN_OUT = "results/example_dnn.wav"

def main():
    assert Path(CLEAN).exists(), "Missing clean audio"
    assert Path(NOISY).exists(), "Missing noisy audio"
    assert Path(DNN_OUT).exists(), "Missing DNN output, run train_mask.py first"

    print(">>> Evaluating DNN")
    print("[NOISY]", eval_pair(CLEAN, NOISY))
    print("[DNN]  ", eval_pair(CLEAN, DNN_OUT))

if __name__ == "__main__":
    main()
