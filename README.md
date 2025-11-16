# ELEC5305 Speech Enhancement Project

Speech enhancement project with classical methods (spectral subtraction, Wiener filtering, IRM) and deep learning models (ImprovedMaskNet, UNet). Features STFT/ISTFT frontend, on-the-fly mixing, objective metrics (SNR/PESQ/STOI), and visualizations.

## Features

- **Classical Methods**: Spectral subtraction, Wiener filtering, Ideal Ratio Mask (IRM)
- **Deep Learning**: ImprovedMaskNet (LSTM-based) and UNet
- **Training**: Learning rate scheduling, early stopping, gradient clipping, AMP support
- **Evaluation**: SNR, PESQ, STOI metrics and visualization tools

## Installation

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (GPU support, CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Data Preparation

### 0. Smoke Test (Verify STFT/ISTFT)

```powershell
python scripts/smoke_test.py
```

Verifies STFT/ISTFT reconstruction. Output: `results/smoke_recon.wav`

### 1. Resample Audio to 16kHz

```powershell
python scripts/resample_all.py
```

Resamples audio from `data/public/` to `data16/` at 16kHz.

### 2. Generate Manifests

**On-the-fly mixing (recommended):**
```powershell
python scripts/make_manifest.py `
  --mode on_the_fly `
  --clean_dir data16/train_clean `
  --noise_dir data16/noise `
  --pairs_per_clean 3 `
  --out_train manifests/train.csv `
  --val_clean_dir data16/val_clean `
  --val_noise_dir data16/noise `
  --out_val manifests/val.csv
```

**Pre-mixed data:**
```powershell
python scripts/make_manifest.py `
  --mode pre_mixed `
  --clean_dir data16/train_clean `
  --noisy_dir data16/train_noisy `
  --out_train manifests/train.csv `
  --val_clean_dir data16/val_clean `
  --val_noisy_dir data16/val_noisy `
  --out_val manifests/val.csv
```

## Classical Methods

### Spectral Subtraction
```powershell
python scripts/noise_test.py
```
Output: `data/noisy/example_noisy.wav`, `results/example_denoised.wav`

### Wiener Filtering
```powershell
python scripts/wiener_test.py
```
Output: `results/example_wiener.wav`

### Ideal Ratio Mask (IRM)
```powershell
python scripts/mask_test.py
```
Output: `results/example_mask_irm.wav`

### Evaluate Classical Methods
```powershell
python scripts/eval_wiener.py
python scripts/eval_mask.py
```
Output: `results/metrics.csv`

## Deep Learning Models

### Train ImprovedMaskNet

```powershell
python scripts/train_mask_improved.py `
  --manifest-train manifests/train.csv `
  --manifest-val manifests/val.csv `
  --mode pre_mixed `
  --model-type improved `
  --use-log `
  --batch 32 `
  --epochs 30 `
  --lr 3e-3 `
  --save-dir checkpoints/improved_gpu `
  --amp `
  --device auto
```

**Key parameters:**
- `--mode pre_mixed`: Faster data loading (recommended)
- `--use-log`: Use log magnitude spectrogram (recommended)
- `--amp`: Enable Automatic Mixed Precision for faster GPU training

**Outputs:**
- `masknet_best.pt`: Best model checkpoint
- `train_log.csv`: Training logs

### Train UNet

```powershell
python scripts/train_unet.py `
  --manifest-train manifests/train.csv `
  --manifest-val manifests/val.csv `
  --save-dir checkpoints/unet `
  --epochs 30 `
  --batch 32 `
  --lr 2e-3 `
  --device auto
```

**Outputs:**
- `unet_best.pt`: Best model checkpoint
- `train_log_unet.csv`: Training logs

## Model Evaluation

### Evaluate ImprovedMaskNet

**Single file:**
```powershell
python scripts/eval_dnn.py `
  --ckpt checkpoints/improved_gpu/masknet_best.pt `
  --clean data/clean/example.wav `
  --noisy data/noisy/example_noisy.wav `
  --outdir results `
  --device auto
```

**Batch evaluation:**
```powershell
python scripts/eval_dnn_batch.py `
  --ckpt checkpoints/improved_gpu/masknet_best.pt `
  --manifest manifests/val.csv `
  --outdir results/batch `
  --device auto
```

**Outputs:** `enhanced_from_ckpt.wav`, `report.csv`

### Evaluate UNet

**Single evaluation:**
```powershell
python scripts/eval_unet.py `
  --ckpt checkpoints/unet/unet_best.pt `
  --clean data/clean/example.wav `
  --noisy data/noisy/example_noisy.wav `
  --outdir results_unet_new `
  --device auto
```

**Outputs:** `enhanced_unet.wav`, `metrics_unet.csv`

**Batch evaluation:**
```powershell
python scripts/eval_unet_batch.py `
  --ckpt checkpoints/unet/unet_best.pt `
  --manifest manifests/val.csv `
  --outdir results/unet_batch `
  --device auto
```

**Outputs:** `enhanced/<stem>_enh.wav`, `all_reports.csv`, `all_reports.json`

## Visualization

### Plot Training Curves

**ImprovedMaskNet:**
```powershell
python scripts/train_plot.py `
  --log checkpoints/improved_gpu/train_log.csv `
  --out checkpoints/improved_gpu/train_curves.png
```

**UNet:**
```powershell
python scripts/train_plot_unet.py `
  --log checkpoints/unet/train_log_unet.csv `
  --out checkpoints/unet/train_curves_unet.png
```

**Compare models:**
```powershell
python scripts/plot_unet_vs_dnn.py `
  --dnn-log checkpoints/improved_gpu/train_log.csv `
  --unet-log checkpoints/unet/train_log_unet.csv `
  --out results/train_curve_unet_vs_dnn.png
```

### Visualize Enhancement Results

```powershell
python scripts/plot_result.py
```

Generates waveform and spectrogram plots for all methods, including comparison plots (Clean/Noisy/DNN/UNet). Outputs saved to `results/plots/`.

## Metrics

- **SNR (dB)**: Signal-to-noise ratio
- **PESQ**: Perceptual evaluation of speech quality (range: -0.5 to 4.5)
- **STOI**: Short-time objective intelligibility (range: 0 to 1)

Higher values indicate better enhancement quality.

## Additional Resources

- `IMPROVEMENTS.md`: Detailed explanation of MaskNet improvements
- `COLAB_SETUP.md`: Guide for training in Google Colab

## Author

Project for ELEC5305 Speech Processing, University of Sydney.  
Maintainer: Zechen Li
