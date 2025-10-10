# ELEC5305 Speech Enhancement Project

This repository contains the implementation of a speech enhancement system using classical filters (Wiener, subtraction) and mask-based deep learning methods.

# Project Structure
speech-enhance-mask/
├── data/ # Folder for input/output audio
│ ├── clean/ # Clean speech samples
│ └── noisy/ # Noisy speech samples
├── results/ # Enhanced and evaluation outputs
├── scripts/ # Main scripts
│ ├── smoke_test.py # Quick test script
│ ├── noise_test.py # Add noise & run baseline
│ ├── eval_wiener.py# Evaluate Wiener filter
│ ├── eval_mask.py # Evaluate mask-based model
│ └── train_mask.py # Train DNN mask model
├── src/ # Source code
│ ├── utils.py # Utility functions
│ ├── stft.py # STFT/ISTFT functions
│ ├── add_noise.py # Noise injection
│ └── dnn_mask.py # MaskNet model
├── requirements.txt # Python dependencies
└── README.md # Project documentation


# 1.Create a virtual environment

python3 -m venv .venv
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

# Metrics
We evaluate using:

SNR (dB) – Signal-to-noise ratio

PESQ – Perceptual evaluation of speech quality

STOI – Short-time objective intelligibility

# Author

Project for ELEC5305 Speech Processing, University of Sydney.
Maintainer: Zechen Li