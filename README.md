# ELEC5305 Speech Enhancement Project

This is a speech enhancement project featuring classical methods (spectral subtraction, Wiener, ideal masks) and a DNN mask model, plus STFT/ISTFT frontend, on-the-fly noisy datasets, objective metrics (SNR/PESQ/STOI) and visualizations.

# Project Structure
.
├─ src/ # core modules
│ ├─ add_noise.py # white-noise injection
│ ├─ dataset.py # PairDataset(on_the_fly / pre_mixed)
│ ├─ dnn_mask.py # simple MLP mask net
│ ├─ eval_metrics.py # SNR、PESQ、STOI / objective metrics
│ ├─ masking.py # ideal ratio/binary masks
│ ├─ stft.py # 16kHz, 25ms, 10ms, NFFT=1024
│ ├─ utils.py # audio utils
│ └─ wiener.py # Wiener filtering
│
├─ scripts/ # experiment scripts
│ ├─ smoke_test.py # STFT↔ISTFT smoke test
│ ├─ noise_test.py # noise + subtraction
│ ├─ wiener_test.py # Wiener
│ ├─ mask_test.py # IRM masking
│ ├─ eval_test.py # basic eval
│ ├─ eval_wiener.py # Wiener eval
│ ├─ eval_mask.py # summary eval
│ ├─ eval_dnn.py # single-file DNN
│ ├─ eval_dnn_batch.py # batch DNN eval
│ ├─ train_mask.py # train (50 epochs + early stop)
│ ├─ plot_result.py # plots
│ └─ make_manifest.py # manifest builder
│
├─ data/
│ ├─ clean/ # demo clean
│ ├─ noisy/ # generated noisy
│ └─ public/ # public subsets
├─ manifests/ # CSV manifests
├─ checkpoints/ # checkpoints
└─ results/ # outputs & plots 


Step 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux  
.venv\Scripts\activate         # Windows  

Step 2. Install dependencies
pip install -r requirements.txt


Windows 用户如需 GPU 训练，请安装对应 CUDA 版本的 PyTorch：

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


macOS 用户默认使用 CPU，如为 M1/M2 芯片可开启 Metal 加速：

pip install torch torchvision torchaudio

1. Run smoke test / 验证环境
python scripts/smoke_test.py

2. Run noise test / 生成加噪音频
python scripts/noise_test.py


执行后将在 results/ 目录下生成加噪与去噪结果。

3. Evaluate models / 传统方法评估
python scripts/eval_wiener.py
python scripts/eval_mask.py


对比 Wiener 与 IRM 方法的增强效果，输出 PESQ/STOI 指标。

4. Train DNN mask model / 训练掩蔽网络
python scripts/train_mask.py ^
  --manifest-train manifests/train.csv ^
  --manifest-val manifests/val.csv ^
  --mode_on_the_fly ^
  --snr-list -5 0 5 10 ^
  --batch 8 ^
  --epochs 50 ^
  --lr 1e-3 ^
  --workers 0 ^
  --save-dir checkpoints/demo ^
  --seed 1337 ^
  --device cuda


训练结果：

masknet_best.pt（验证集最优）

masknet_last.pt（最后一轮）

train_log.csv（训练日志，记录 BCE 损失）

5. Evaluate DNN model / 模型推理评估
python scripts/eval_dnn.py ^
  --ckpt checkpoints/demo/masknet_best.pt ^
  --clean data/clean/example.wav ^
  --noisy data/noisy/example_noisy.wav ^
  --outdir results ^
  --device auto



output：

enhanced_from_ckpt.wav: Enhanced speech
report.csv / report.json：Indicator Report

6. Plot results /Plotting waveforms and spectrum diagrams
python scripts/plot_result.py

Automatically generate waveforms and spectra of all enhancement results and save them to [location]：
results/plots/

7. Plot training curves / Plotting training curves
python scripts/train_plot.py \
  --log checkpoints/demo/train_log.csv \
  --out checkpoints/demo/train_curves.png

Output: Training and validation BCE loss curves.

Metrics

We evaluate using:

SNR (dB) – Signal-to-noise ratio

PESQ – Perceptual evaluation of speech quality

STOI – Short-time objective intelligibility

# Author

Project for ELEC5305 Speech Processing, University of Sydney.
Maintainer: Zechen Li