from PIL import Image
import os

os.makedirs("results/grids", exist_ok=True)

n = 20
w, h = 512, 512
grid = Image.new("RGB", (2*w, n*h))

for i in range(n):
    inp = Image.open(f"results/inputs/in_{i}.jpg").convert("RGB").resize((w,h))
    rec = Image.open(f"results/recon/recon_{i}.jpg").convert("RGB").resize((w,h))
    grid.paste(inp, (0, i*h))
    grid.paste(rec, (w, i*h))

grid.save("results/grids/sanity_input_vs_recon.jpg")
print("Saved: results/grids/sanity_input_vs_recon.jpg")
