import os
import sys
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
STYLE_CONFIG = {
    'colors': {
        'dnn_train': '#1f77b4',
        'dnn_val': '#aec7e8',
        'unet_train': '#d62728',
        'unet_val': '#ff9896',
    },
    'linewidth': 3.5,
    'linewidth_dashed': 2.5,
    'markersize': 6,
    'markeredgewidth': 1.5,
    'markeredgecolor': 'white',
    'fontsize': {
        'title': 16,
        'label': 14,
        'legend': 12,
        'tick': 12,
    },
    'fontweight': 'bold',
    'grid_alpha': 0.4,
    'grid_linewidth': 1.2,
    'legend_framealpha': 0.9,
    'figsize': (18, 6),
    # DPI
    'dpi': 200,
}

"""
Plot training curves for both models:
- DNN: train_log.csv
- UNet: train_log_unet.csv

Output: results/train_curve_unet_vs_dnn.png
"""

def load_curve(path: Path, key_train: str, key_val: str):
    assert path.exists(), f"CSV file not found: {path}"
    epochs, tr, va = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            tr.append(float(row[key_train]))
            v = row.get(key_val, "")
            if v == "" or v is None:
                va.append(float("nan"))
            else:
                va.append(float(v))
    return epochs, tr, va


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Plot UNet vs DNN training curves")
    ap.add_argument("--dnn-log", type=str, default="checkpoints/demo/train_log.csv",
                    help="DNN training log (from train_mask)")
    ap.add_argument("--unet-log", type=str, default="checkpoints/unet/train_log_unet.csv",
                    help="UNet training log")
    ap.add_argument("--out", type=str, default="results/train_curve_unet_vs_dnn.png")
    args = ap.parse_args()

    dnn_path = Path(args.dnn_log)
    unet_path = Path(args.unet_log)

    dnn_epochs, dnn_tr, dnn_va = load_curve(
        dnn_path, key_train="train_bce", key_val="val_bce"
    )

    unet_epochs, unet_tr, unet_va = load_curve(
        unet_path, key_train="train_loss", key_val="val_loss"
    )

    fig, axes = plt.subplots(1, 2, figsize=STYLE_CONFIG['figsize'])
    
    ax1 = axes[0]
    ax1.plot(dnn_epochs, dnn_tr, label="DNN Train (BCE)",
             linestyle="-",
             linewidth=STYLE_CONFIG['linewidth'],
             marker='o',
             markersize=STYLE_CONFIG['markersize'],
             color=STYLE_CONFIG['colors']['dnn_train'],
             markerfacecolor=STYLE_CONFIG['colors']['dnn_train'],
             markeredgewidth=STYLE_CONFIG['markeredgewidth'],
             markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    if any(x == x for x in dnn_va):
        ax1.plot(dnn_epochs, [x if x==x else None for x in dnn_va], label="DNN Val (BCE)",
                 linestyle="--",
                 linewidth=STYLE_CONFIG['linewidth_dashed'],
                 marker='s',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['dnn_val'],
                 markerfacecolor=STYLE_CONFIG['colors']['dnn_val'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    ax1.plot(unet_epochs, unet_tr, label="UNet Train (L1)",
             linestyle="-",
             linewidth=STYLE_CONFIG['linewidth'],
             marker='^',
             markersize=STYLE_CONFIG['markersize'],
             color=STYLE_CONFIG['colors']['unet_train'],
             markerfacecolor=STYLE_CONFIG['colors']['unet_train'],
             markeredgewidth=STYLE_CONFIG['markeredgewidth'],
             markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    if any(x == x for x in unet_va):
        ax1.plot(unet_epochs, [x if x==x else None for x in unet_va], label="UNet Val (L1)",
                 linestyle="--",
                 linewidth=STYLE_CONFIG['linewidth_dashed'],
                 marker='v',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['unet_val'],
                 markerfacecolor=STYLE_CONFIG['colors']['unet_val'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    ax1.set_xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
    ax1.set_ylabel("Loss", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
    ax1.set_title("Training Curves: UNet vs DNN", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
    ax1.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
    ax1.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
    ax1.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    
    ax2 = axes[1]
    has_dnn_val = any(x == x for x in dnn_va)
    has_unet_val = any(x == x for x in unet_va)
    
    if has_dnn_val and has_unet_val:
        ax2.plot(dnn_epochs, [x if x==x else None for x in dnn_va], label="DNN Val (BCE)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='s',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['dnn_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['dnn_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.plot(unet_epochs, [x if x==x else None for x in unet_va], label="UNet Val (L1)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='v',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['unet_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['unet_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.set_xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_ylabel("Validation Loss", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_title("Validation Loss Comparison", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
        ax2.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
        ax2.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    elif has_dnn_val:
        ax2.plot(dnn_epochs, [x if x==x else None for x in dnn_va], label="DNN Val (BCE)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='s',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['dnn_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['dnn_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.set_xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_ylabel("Validation Loss", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_title("DNN Validation Loss", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
        ax2.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
        ax2.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    elif has_unet_val:
        ax2.plot(unet_epochs, [x if x==x else None for x in unet_va], label="UNet Val (L1)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='v',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['unet_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['unet_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.set_xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_ylabel("Validation Loss", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_title("UNet Validation Loss", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
        ax2.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
        ax2.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    else:
        ax2.plot(dnn_epochs, dnn_tr, label="DNN Train (BCE)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='o',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['dnn_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['dnn_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.plot(unet_epochs, unet_tr, label="UNet Train (L1)",
                 linestyle="-",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='^',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['unet_train'],
                 markerfacecolor=STYLE_CONFIG['colors']['unet_train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        ax2.set_xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_ylabel("Training Loss", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.set_title("Training Loss Comparison", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        ax2.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
        ax2.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
        ax2.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=STYLE_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[Saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()
