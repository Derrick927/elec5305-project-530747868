import os
import soundfile as sf

clean_dir = "data16/train_clean"
noisy_dir = "data16/train_noisy"

clean_files = sorted(os.listdir(clean_dir))
noisy_files = sorted(os.listdir(noisy_dir))

count_mismatch = 0

for c, n in zip(clean_files, noisy_files):
    cw, sr1 = sf.read(os.path.join(clean_dir, c))
    nw, sr2 = sf.read(os.path.join(noisy_dir, n))

    if len(cw) != len(nw):
        print(f"❌ Length mismatch: {c}, {n}, clean={len(cw)}, noisy={len(nw)}")
        count_mismatch += 1

print(f"\nTotal mismatches: {count_mismatch}")
