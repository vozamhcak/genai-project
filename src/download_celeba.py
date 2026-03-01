import os
import shutil
from torchvision.datasets import CelebA

ROOT = "data/celeba_download"
OUT_DIR = "data/celeba_20pct"
FRAC = 0.2

os.makedirs(ROOT, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ds = CelebA(root=ROOT, split="train", download=True)
n = len(ds)
k = max(1, int(n * FRAC))
step = max(1, n // k)
indices = list(range(0, n, step))[:k]

src_dir = os.path.join(ROOT, "celeba", "img_align_celeba")
for i in indices:
    fn = ds.filename[i]
    src = os.path.join(src_dir, fn)
    dst = os.path.join(OUT_DIR, fn)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

print(f"{len(indices)} images -> {OUT_DIR}")
