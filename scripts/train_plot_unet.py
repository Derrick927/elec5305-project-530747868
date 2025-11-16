import os, sys, csv
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
STYLE_CONFIG = {
    'colors': {
        'train': '#1f77b4',
        'val': '#d62728',
        'lr': '#2ca02c',
        'diff': '#9467bd',
    },
    'linewidth': 3.5,
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
    'figsize_single': (12, 6),
    'figsize_multi': (18, 6),
    # DPI
    'dpi': 200,
}

def parse_args():
    ap = argparse.ArgumentParser(description="Plot UNet training curves from CSV log.")
    ap.add_argument("--log", type=str, required=True, help="Path to train_log_unet.csv")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: alongside log)")
    return ap.parse_args()

def main():
    args = parse_args()
    log_path = Path(args.log)
    assert log_path.exists(), f"CSV not found: {log_path}"

    epochs, tr, va, lrs = [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            tr.append(float(row["train_loss"]))
            v = row.get("val_loss", "")
            va.append(float(v) if v not in ("", None) else float("nan"))
            lr_str = row.get("lr", "")
            if lr_str and lr_str not in ("", None):
                lrs.append(float(lr_str))
            else:
                    lrs.append(float("nan"))

    has_lr = any(x == x for x in lrs)
    has_val = any(x == x for x in va)
    n_subplots = 3 if (has_lr and has_val) else (2 if (has_lr or has_val) else 1)
    
    plt.figure(figsize=STYLE_CONFIG['figsize_multi'] if n_subplots > 1 else STYLE_CONFIG['figsize_single'])
    
    plt.subplot(1, n_subplots, 1)
    plt.plot(epochs, tr, label="Train Loss",
             linewidth=STYLE_CONFIG['linewidth'],
             marker='o',
             markersize=STYLE_CONFIG['markersize'],
             color=STYLE_CONFIG['colors']['train'],
             markerfacecolor=STYLE_CONFIG['colors']['train'],
             markeredgewidth=STYLE_CONFIG['markeredgewidth'],
             markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    if has_val:
        plt.plot(epochs, [x if x==x else None for x in va], label="Val Loss",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='s',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['val'],
                 markerfacecolor=STYLE_CONFIG['colors']['val'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
    
    plt.xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
    plt.ylabel("Loss (L1)", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
    plt.title("UNet Training Curves", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
    plt.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
    plt.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
    plt.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    
    if has_lr:
        plt.subplot(1, n_subplots, 2)
        plt.plot(epochs, [x if x==x else None for x in lrs], label="Learning Rate",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='^',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['lr'],
                 markerfacecolor=STYLE_CONFIG['colors']['lr'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        plt.xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        plt.ylabel("Learning Rate", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        plt.title("Learning Rate Schedule", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        plt.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
    plt.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
    plt.yscale('log')
    plt.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    
    if has_val:
        subplot_idx = 3 if has_lr else 2
        plt.subplot(1, n_subplots, subplot_idx)
        diff = [t - v if (v==v) else None for t, v in zip(tr, va)]
        valid_diff = [d for d in diff if d is not None]
        valid_epochs = [e for e, d in zip(epochs, diff) if d is not None]
        if valid_diff:
            plt.plot(valid_epochs, valid_diff, label="Train - Val",
                     linewidth=STYLE_CONFIG['linewidth'],
                     color=STYLE_CONFIG['colors']['diff'],
                     marker='^',
                     markersize=STYLE_CONFIG['markersize'],
                     markerfacecolor=STYLE_CONFIG['colors']['diff'],
                     markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                     markeredgecolor=STYLE_CONFIG['markeredgecolor'])
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2, label="Zero Line")
            plt.xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
            plt.ylabel("Loss Difference", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
            plt.title("Overfitting Indicator", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
            plt.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
            plt.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
            plt.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    elif not has_lr:
        plt.subplot(1, n_subplots, 2)
        plt.plot(epochs, tr, label="Train Loss",
                 linewidth=STYLE_CONFIG['linewidth'],
                 marker='o',
                 markersize=STYLE_CONFIG['markersize'],
                 color=STYLE_CONFIG['colors']['train'],
                 markerfacecolor=STYLE_CONFIG['colors']['train'],
                 markeredgewidth=STYLE_CONFIG['markeredgewidth'],
                 markeredgecolor=STYLE_CONFIG['markeredgecolor'])
        plt.xlabel("Epoch", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        plt.ylabel("Loss (L1)", fontsize=STYLE_CONFIG['fontsize']['label'], fontweight=STYLE_CONFIG['fontweight'])
        plt.title("Training Loss Only", fontsize=STYLE_CONFIG['fontsize']['title'], fontweight=STYLE_CONFIG['fontweight'])
        plt.legend(fontsize=STYLE_CONFIG['fontsize']['legend'], loc='best', framealpha=STYLE_CONFIG['legend_framealpha'])
        plt.grid(True, alpha=STYLE_CONFIG['grid_alpha'], linewidth=STYLE_CONFIG['grid_linewidth'])
        plt.tick_params(labelsize=STYLE_CONFIG['fontsize']['tick'])
    
    plt.tight_layout()

    out = Path(args.out) if args.out else (log_path.parent / "train_curves_unet.png")
    plt.savefig(out, dpi=STYLE_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[Saved] {out.resolve()}")

if __name__ == "__main__":
    main()

