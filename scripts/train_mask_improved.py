import os, math, json, argparse, random
import warnings
import signal
import sys
from pathlib import Path
from typing import Optional, List

# Suppress warnings before importing other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Make src importable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import PairDataset
from src.dnn_mask_improved import MaskNet, ImprovedMaskNet, DeepMaskNet
from src.stft import N_FFT

def seed_all(seed: int = 1337):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_dirs(path: str): Path(path).mkdir(parents=True, exist_ok=True)

def collate_pad(batch):
    Ts = [x[0].shape[0] for x in batch]
    Fdim = batch[0][0].shape[1]; T_max = max(Ts); B = len(batch)
    feats = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    labels = torch.zeros((B, T_max, Fdim), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.float32)
    for i, (noisy_T, irm_T) in enumerate(batch):
        t = noisy_T.shape[0]
        feats[i, :t, :] = torch.from_numpy(noisy_T)
        labels[i, :t, :] = torch.from_numpy(irm_T)
        mask[i, :t] = 1.0
    return feats, labels, mask

def masked_bce_loss(pred, target, mask, use_logits=False):
    """
    Args:
        pred: predictions (logits if use_logits=True, else [0,1] after sigmoid)
        target: targets in [0,1]
        mask: valid frame mask
        use_logits: if True, pred are logits (use BCEWithLogitsLoss), else use BCELoss
    """
    if use_logits:
        bce = nn.BCEWithLogitsLoss(reduction="none")
        loss = bce(pred, target) * mask.unsqueeze(-1)  # (B, T, F)
    else:
        bce = nn.BCELoss(reduction="none")
        loss = bce(pred, target) * mask.unsqueeze(-1)  # (B, T, F)
    valid_elems = mask.sum() * pred.size(-1)
    return loss.sum() / (valid_elems + 1e-8)

def masked_mse_loss(pred, target, mask):
    mse = nn.MSELoss(reduction="none")
    loss = mse(pred, target) * mask.unsqueeze(-1)
    valid_elems = mask.sum() * pred.size(-1)
    return loss.sum() / (valid_elems + 1e-8)

@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device, use_logits=False) -> float:
    model.eval(); total_loss = 0.0; total_frames = 0.0
    for feats, labels, mask in loader:
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        if use_logits:
            model._return_logits = True
            preds = model(feats)
            loss = masked_bce_loss(preds, labels, mask, use_logits=True)
            model._return_logits = False
        else:
            preds = model(feats)
            loss = masked_bce_loss(preds, labels, mask, use_logits=False)
        frames = mask.sum().item()
        total_loss += loss.item() * frames; total_frames += frames
    return total_loss / max(total_frames, 1.0)

def normalize_snr_list(val) -> Optional[List[float]]:
    if val is None: return None
    if isinstance(val, list):
        parts = [p.strip() for p in (val[0].split(",") if (len(val)==1 and "," in val[0]) else val)]
    else:
        parts = [p.strip() for p in str(val).split(",")]
    parts = [p for p in parts if p]; 
    return [float(p) for p in parts] if parts else None


def parse_args():
    ap = argparse.ArgumentParser(description="Train improved mask-based enhancement (IRM).")
    ap.add_argument("--manifest-train", required=True)
    ap.add_argument("--manifest-val", default="")
    ap.add_argument("--mode", default="on_the_fly", choices=["on_the_fly","pre_mixed"])
    ap.add_argument("--model-type", default="improved", choices=["original", "improved", "deep"],
                    help="Model architecture: original=simple MLP, improved=LSTM-based, deep=deep MLP")
    ap.add_argument("--batch", type=int, default=32, help="Batch size (default: 32 for GPU training)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-3, help="Learning rate (default: 3e-3 for faster convergence)")
    ap.add_argument("--warmup-epochs", type=int, default=2, help="Warmup epochs (default: 2, shorter for faster start)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4 for GPU training)")
    ap.add_argument("--save-dir", default="checkpoints/improved")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda", choices=["auto","cpu","cuda"],
                    help="Device for training (default: cuda for GPU training)")
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="Early stopping patience (number of epochs without improvement)")
    ap.add_argument("--early-stop-delta", type=float, default=0.005,
                    help="Minimum improvement threshold for early stopping (default: 0.005)")
    ap.add_argument("--snr-min", type=float, default=None)
    ap.add_argument("--snr-max", type=float, default=None)
    ap.add_argument("--snr-list", nargs="+", default=None)
    ap.add_argument("--noise-filter", type=str, default="")
    ap.add_argument("--use-log", action="store_true", help="Use log magnitude spectrogram")
    ap.add_argument("--loss-type", default="combined", choices=["bce", "mse", "combined"],
                    help="Loss function type (default: combined for better learning)")
    ap.add_argument("--amp", action="store_true", help="Use automatic mixed precision (recommended for GPU, default: auto-enabled on GPU)")
    ap.add_argument("--no-amp", action="store_true", dest="no_amp", help="Disable automatic mixed precision")
    return ap.parse_args()

def resolve_device(flag: str) -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if flag == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    args = parse_args()
    seed_all(args.seed); make_dirs(args.save_dir)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"[Info] Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"[Info] CUDA enabled optimizations: cudnn.benchmark=True")
    else:
        print(f"[Info] Device: {device} (CPU training - consider using GPU for faster training)")
    print(f"[Info] Model type: {args.model_type}")

    snr_list = normalize_snr_list(args.snr_list)
    in_dim = N_FFT // 2 + 1

    ds_train = PairDataset(args.manifest_train, mode=args.mode,
                           snr_min=args.snr_min, snr_max=args.snr_max,
                           snr_list=snr_list, noise_filter=(args.noise_filter or None),
                           seed=args.seed)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate_pad)

    dl_val = None
    if args.manifest_val:
        fixed_snr_val = None
        if args.mode == "on_the_fly":
            if snr_list and len(snr_list) > 0:
                fixed_snr_val = snr_list[len(snr_list) // 2]
            elif args.snr_min is not None and args.snr_max is not None:
                fixed_snr_val = (args.snr_min + args.snr_max) / 2.0
            else:
                fixed_snr_val = 0.0
        
        ds_val = PairDataset(args.manifest_val, mode=args.mode,
                             snr_min=args.snr_min, snr_max=args.snr_max,
                             snr_list=snr_list, noise_filter=(args.noise_filter or None),
                             seed=args.seed+1, fixed_snr=fixed_snr_val)
        dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_pad)
        if fixed_snr_val is not None:
            print(f"[Info] Validation set uses fixed SNR: {fixed_snr_val:.1f} dB")

    if args.model_type == "original":
        model = MaskNet(in_dim=in_dim).to(device)
    elif args.model_type == "improved":
        model = ImprovedMaskNet(in_dim=in_dim, use_log=args.use_log).to(device)
    elif args.model_type == "deep":
        model = DeepMaskNet(in_dim=in_dim, use_log=args.use_log).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    optimiz = optim.Adam(model.parameters(), lr=args.lr, 
                        betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiz, mode='min', factor=0.5, patience=6, min_lr=1e-6)
    
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimiz, 
            lr_lambda=lambda epoch: (epoch + 1) / args.warmup_epochs if epoch < args.warmup_epochs else 1.0
        )

    run_meta = {
        "mode": args.mode, "manifest_train": args.manifest_train,
        "manifest_val": args.manifest_val or None, "save_dir": args.save_dir,
        "seed": args.seed, "snr_min": args.snr_min, "snr_max": args.snr_max,
        "snr_list": snr_list, "noise_filter": args.noise_filter or None,
        "batch": args.batch, "epochs": args.epochs, "lr": args.lr,
        "workers": args.workers, "in_dim": in_dim,
        "train_pairs": len(ds_train), "val_pairs": (len(ds_val) if dl_val else 0),
        "device": str(device), "model_type": args.model_type,
        "use_log": args.use_log, "loss_type": args.loss_type,
        "early_stop_patience": (args.early_stop_patience if dl_val else None),
        "early_stop_delta": (args.early_stop_delta if dl_val else None),
        "total_params": total_params,
    }
    meta_path = Path(args.save_dir) / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f: json.dump(run_meta, f, ensure_ascii=False, indent=2)
    print(f"[Info] Saved run metadata -> {meta_path}")

    log_path = Path(args.save_dir) / "train_log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_bce,val_bce,lr\n")

    best_val = float("inf")
    best_epoch = -1
    patience_cnt = 0
    min_delta = args.early_stop_delta
    best_path = Path(args.save_dir) / "masknet_best.pt"
    
    if dl_val:
        print(f"[Info] Early stopping: patience={args.early_stop_patience}, min_delta={min_delta}")
    
    use_amp = (args.amp or (not args.no_amp and device.type == "cuda")) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("[Info] Using Automatic Mixed Precision (AMP) for faster GPU training")
    elif device.type == "cuda":
        print("[Info] AMP disabled - consider enabling with --amp for faster training")

    current_epoch_state = {"epoch": 0, "tr_loss": None, "val_loss": None, "current_lr": None}
    
    def save_log_and_exit(signum, frame):
        """Handle interrupt signal and save current state"""
        print("\n[Interrupted] Saving training log and model...")
        if current_epoch_state["epoch"] > 0:
            with open(log_path, "a", encoding="utf-8") as f:
                val_str = current_epoch_state["val_loss"] if current_epoch_state["val_loss"] is not None else ""
                f.write(f"{current_epoch_state['epoch']},{current_epoch_state['tr_loss']},{val_str},{current_epoch_state['current_lr']}\n")
            last_path = Path(args.save_dir) / "masknet_last.pt"
            torch.save({
                "epoch": current_epoch_state["epoch"],
                "state_dict": model.state_dict(),
                "val_bce": current_epoch_state["val_loss"],
                "run_meta": run_meta
            }, last_path)
            print(f"[Saved] Training log and model checkpoint saved.")
        print("[Done] Training interrupted by user.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, save_log_and_exit)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, save_log_and_exit)

    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0; seen = 0.0
        for feats, labels, mask in dl_train:
            feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
            
            optimiz.zero_grad(set_to_none=True)
            
            if use_amp:
                model._return_logits = True
                with autocast():
                    preds_logits = model(feats)  # logits
                    if args.loss_type == "bce":
                        loss = masked_bce_loss(preds_logits, labels, mask, use_logits=True)
                    elif args.loss_type == "mse":
                        preds = torch.sigmoid(preds_logits)
                        loss = masked_mse_loss(preds, labels, mask)
                    elif args.loss_type == "combined":
                        preds = torch.sigmoid(preds_logits)
                        loss = 0.7 * masked_bce_loss(preds_logits, labels, mask, use_logits=True) + 0.3 * masked_mse_loss(preds, labels, mask)
                    else:
                        loss = masked_bce_loss(preds_logits, labels, mask, use_logits=True)
                model._return_logits = False
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimiz)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimiz)
                scaler.update()
            else:
                preds = model(feats)
                if args.loss_type == "bce":
                    loss = masked_bce_loss(preds, labels, mask)
                elif args.loss_type == "mse":
                    loss = masked_mse_loss(preds, labels, mask)
                elif args.loss_type == "combined":
                    loss = 0.7 * masked_bce_loss(preds, labels, mask) + 0.3 * masked_mse_loss(preds, labels, mask)
                else:
                    loss = masked_bce_loss(preds, labels, mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimiz.step()
            
            frames = mask.sum().item(); running += loss.item()*frames; seen += frames
        tr_loss = running / max(seen, 1.0)

        if dl_val:
            val_loss = run_eval(model, dl_val, device, use_logits=use_amp)
            current_lr = optimiz.param_groups[0]['lr']
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}  val_bce={val_loss:.6f}  lr={current_lr:.2e}")
            
            if warmup_scheduler is not None and epoch <= args.warmup_epochs:
                warmup_scheduler.step()
                current_lr = optimiz.param_groups[0]['lr']
                if epoch == args.warmup_epochs:
                    print(f"[Info] Warmup finished, switching to ReduceLROnPlateau scheduler")
            else:
                scheduler.step(val_loss)
                current_lr = optimiz.param_groups[0]['lr']
            
            if best_val - val_loss > min_delta:
                best_val = val_loss
                best_epoch = epoch
                patience_cnt = 0
                torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                            "val_bce": best_val, "run_meta": run_meta}, best_path)
                print(f"[Best] Saved best model with val_bce={best_val:.6f}")
            else:
                patience_cnt += 1
                if val_loss < best_val:
                    best_val = val_loss
                    best_epoch = epoch
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                                "val_bce": best_val, "run_meta": run_meta}, best_path)
                print(f"[Info] No significant improvement, patience = {patience_cnt}/{args.early_stop_patience}")

                if patience_cnt >= args.early_stop_patience:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{epoch},{tr_loss},{val_loss},{current_lr}\n")
                    last_path = Path(args.save_dir) / "masknet_last.pt"
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                                "val_bce": val_loss, "run_meta": run_meta}, last_path)
                    print(f"[EarlyStop] Patience reached. Best val_bce={best_val:.6f} at epoch {best_epoch}.")
                    print("[Done] Training finished (early stopped).")
                    return
        else:
            val_loss = None
            current_lr = optimiz.param_groups[0]['lr']
            print(f"[Epoch {epoch:03d}] train_bce={tr_loss:.6f}  lr={current_lr:.2e}")

        current_epoch_state["epoch"] = epoch
        current_epoch_state["tr_loss"] = tr_loss
        current_epoch_state["val_loss"] = val_loss
        current_epoch_state["current_lr"] = current_lr

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr_loss},{'' if val_loss is None else val_loss},{current_lr}\n")

        last_path = Path(args.save_dir) / "masknet_last.pt"
        torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                    "val_bce": val_loss, "run_meta": run_meta}, last_path)

    if dl_val:
        print(f"[Done] best epoch={best_epoch}, min val_bce={best_val:.6f}")
    else:
        print("[Done] Training finished.")

if __name__ == "__main__":
    main()

