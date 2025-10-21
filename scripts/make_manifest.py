import argparse
import random
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac"}

def list_audio(dir_path: str):
    p = Path(dir_path) if dir_path else None
    if not p:
        return []
    return sorted([str(x) for x in p.rglob("*") if x.suffix.lower() in AUDIO_EXTS])

def write_lines(lines, out_path: str):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for a, b in lines:
            f.write(f"{a},{b}\n")
    print(f"[OK] Wrote {len(lines)} pairs to: {out}")

def pair_on_the_fly(clean_list, noise_list, pairs_per_clean: int = 3, seed=1337):
    """
    Build (clean, noise) pairs. We randomly sample 'pairs_per_clean' noises for each clean file.
    """
    assert len(clean_list) > 0, "Empty clean list."
    assert len(noise_list) > 0, "Empty noise list."
    rnd = random.Random(seed)
    pairs = []
    for c in clean_list:
        for _ in range(pairs_per_clean):
            n = rnd.choice(noise_list)
            pairs.append((c, n))
    return pairs

def pair_premixed(clean_list, noisy_list):
    """
    Build (clean, noisy) pairs by filename stem matching if possible; 
    otherwise pair them in sorted order length-wise.
    """
    if len(clean_list) == 0 or len(noisy_list) == 0:
        raise ValueError("Empty clean or noisy list for pre_mixed mode.")
    # Try stem matching
    by_stem_clean = {Path(p).stem: p for p in clean_list}
    by_stem_noisy = {Path(p).stem: p for p in noisy_list}
    common = sorted(set(by_stem_clean.keys()) & set(by_stem_noisy.keys()))
    pairs = []
    if common:
        for s in common:
            pairs.append((by_stem_clean[s], by_stem_noisy[s]))
    else:
        # Fallback: zip by sorted order
        L = min(len(clean_list), len(noisy_list))
        pairs = list(zip(clean_list[:L], noisy_list[:L]))
    return pairs

def build(mode: str,
          clean_dir: str,
          noise_dir: str,
          noisy_dir: str,
          pairs_per_clean: int):
    clean_list = list_audio(clean_dir)
    if mode == "on_the_fly":
        noise_list = list_audio(noise_dir)
        return pair_on_the_fly(clean_list, noise_list, pairs_per_clean=pairs_per_clean)
    else:
        noisy_list = list_audio(noisy_dir)
        return pair_premixed(clean_list, noisy_list)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["on_the_fly", "pre_mixed"],
                    help="How to build pairs for PairDataset.")
    ap.add_argument("--clean_dir", type=str, required=True, help="Folder of clean wavs.")
    # on_the_fly inputs
    ap.add_argument("--noise_dir", type=str, default="", help="Folder of noise wavs (on_the_fly).")
    ap.add_argument("--pairs_per_clean", type=int, default=3,
                    help="How many (clean,noise) pairs per clean sample for on_the_fly.")
    # pre_mixed inputs
    ap.add_argument("--noisy_dir", type=str, default="", help="Folder of pre-mixed noisy wavs (pre_mixed).")
    # validation (optional)
    ap.add_argument("--val_clean_dir", type=str, default="", help="Val clean folder.")
    ap.add_argument("--val_noise_dir", type=str, default="", help="Val noise folder (on_the_fly).")
    ap.add_argument("--val_noisy_dir", type=str, default="", help="Val noisy folder (pre_mixed).")
    # outputs
    ap.add_argument("--out_train", type=str, required=True, help="Output CSV for training.")
    ap.add_argument("--out_val", type=str, default="", help="Output CSV for validation.")
    args = ap.parse_args()

    # Build train
    train_pairs = build(args.mode, args.clean_dir, args.noise_dir, args.noisy_dir, args.pairs_per_clean)
    write_lines(train_pairs, args.out_train)

    # Build val (optional)
    if args.out_val:
        if args.mode == "on_the_fly":
            val_pairs = build(args.mode, args.val_clean_dir, args.val_noise_dir, "", args.pairs_per_clean)
        else:
            val_pairs = build(args.mode, args.val_clean_dir, "", args.val_noisy_dir, args.pairs_per_clean)
        write_lines(val_pairs, args.out_val)

if __name__ == "__main__":
    main()
