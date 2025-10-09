# ELEC5305 Speech Enhancement Project

This repository contains the implementation of a speech enhancement system using classical filters (Wiener, subtraction) and mask-based deep learning methods.

## 📂 Project Structure
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

## 🚀 Quick Start


1.Create a virtual environment

python -m venv .venv
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows
Install dependencies

pip install -r requirements.txt


Run smoke test

python scripts/smoke_test.py


Run noise test

python scripts/noise_test.py


Evaluate models

python scripts/eval_wiener.py
python scripts/eval_mask.py


Train DNN mask model (optional)

python scripts/train_mask.py
We evaluate using:

SNR (dB) – Signal-to-noise ratio

PESQ – Perceptual evaluation of speech quality

STOI – Short-time objective intelligibility

👨‍💻 Author

Project for ELEC5305 Speech Processing, University of Sydney.
Maintainer: Derrick927