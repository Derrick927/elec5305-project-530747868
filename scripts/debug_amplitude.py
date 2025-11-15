import os
import soundfile as sf
import numpy as np

dir_clean = "data16/train_clean"
dir_noisy = "data16/train_noisy"

clean_files = sorted(os.listdir(dir_clean))

for i in range(5):
    cfile = clean_files[i]
    cw, sr = sf.read(os.path.join(dir_clean, cfile))
    print(f"{cfile} -> min/max = {cw.min()}, {cw.max()}")
