import os
from pathlib import Path
import soundfile as sf
import librosa

TARGET_SR = 16000

def resample_and_save(src_path, dst_path):
    wav, sr = sf.read(src_path)
    wav = wav.astype("float32")
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    sf.write(dst_path, wav, TARGET_SR)

def process_folder(src_folder, dst_folder):
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    for wav_file in sorted(src.glob("*.wav")):
        out_path = dst / wav_file.name
        resample_and_save(str(wav_file), str(out_path))

    print(f"[Done] {src_folder} → {dst_folder}")

def main():
    mapping = {
        "data/public/train_clean": "data16/train_clean",
        "data/public/train_noisy": "data16/train_noisy",
        "data/public/val_clean": "data16/val_clean",
        "data/public/val_noisy": "data16/val_noisy",
    }

    for src, dst in mapping.items():
        process_folder(src, dst)

if __name__ == "__main__":
    main()
