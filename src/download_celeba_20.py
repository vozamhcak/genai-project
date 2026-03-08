import os
import shutil
from torchvision.datasets import CelebA

ROOT = "data/celeba_download"
OUT_DIR = "data/faces"
N_IMAGES = 20

os.makedirs(ROOT, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ds = CelebA(root=ROOT, split="train", download=True)

src_dir = os.path.join(ROOT, "celeba", "img_align_celeba")
total = len(ds)

step = max(1, total // N_IMAGES)
indices = list(range(0, total, step))[:N_IMAGES]

copied = 0
for i, idx in enumerate(indices):
    fn = ds.filename[idx]
    src = os.path.join(src_dir, fn)
    dst = os.path.join(OUT_DIR, f"face_{i:02d}.jpg")

    if os.path.isfile(src):
        shutil.copy2(src, dst)
        copied += 1

print(f"Copied {copied} images to {OUT_DIR}")