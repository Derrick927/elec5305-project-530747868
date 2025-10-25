# ELEC5305 Speech Enhancement Project

This is a speech enhancement project featuring classical methods (spectral subtraction, Wiener, ideal masks) and a DNN mask model, plus STFT/ISTFT frontend, on-the-fly noisy datasets, objective metrics (SNR/PESQ/STOI) and visualizations.

# Project Structure
.
├─ src/                       # core modules
│  ├─ add_noise.py            # white-noise injection
│  ├─ dataset.py              # PairDataset(on_the_fly / pre_mixed)
│  ├─ dnn_mask.py             # simple MLP mask net
│  ├─ eval_metrics.py         # SNR、PESQ、STOI / objective metrics
│  ├─ masking.py              # ideal ratio/binary masks
│  ├─ stft.py                 # 16kHz, 25ms, 10ms, NFFT=1024
│  ├─ utils.py                # audio utils
│  └─ wiener.py               # Wiener filtering
│
├─ scripts/                   # experiment scripts
│  ├─ smoke_test.py           # STFT↔ISTFT smoke test
│  ├─ noise_test.py           # noise + subtraction
│  ├─ wiener_test.py          # Wiener
│  ├─ mask_test.py            # IRM masking
│  ├─ eval_test.py            # basic eval
│  ├─ eval_wiener.py          # Wiener eval
│  ├─ eval_mask.py            # summary eval
│  ├─ eval_dnn.py             # single-file DNN 
│  ├─ eval_dnn_batch.py       # batch DNN eval
│  ├─ train_mask.py           # train (50 epochs + early stop )
│  ├─ plot_result.py          # plots
│  └─ make_manifest.py        # manifest builder
│
├─ data/
│  ├─ clean/                  # demo clean
│  ├─ noisy/                  # generated noisy
│  └─ public/                 # public subsets
├─ manifests/                 # CSV manifests
├─ checkpoints/               # checkpoints
└─ results/                   # outputs & plots



# 1.Create a virtual environment

python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows

# 2.Install dependencies

pip install -r requirements.txt

# 3.Run smoke test

python scripts/smoke_test.py

# 4.Run noise test

python scripts/noise_test.py

# 5.Evaluate models

python scripts/eval_wiener.py
python scripts/eval_mask.py

# 6.Train DNN mask model 

python scripts/train_mask.py
python scripts/eval_dnn.py

# Metrics
We evaluate using:

SNR (dB) – Signal-to-noise ratio

PESQ – Perceptual evaluation of speech quality

STOI – Short-time objective intelligibility

# Author

Project for ELEC5305 Speech Processing, University of Sydney.
Maintainer: Zechen Li