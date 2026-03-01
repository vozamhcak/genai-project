from PIL import Image
import os

os.makedirs("results/grids", exist_ok=True)

n = 20  # сколько пар
cols = 2  # input | edit
w, h = 512, 512

grid = Image.new("RGB", (cols * w, n * h))

for i in range(n):
    inp = Image.open(f"results/inputs/in_{i}.jpg").convert("RGB").resize((w, h))
    out = Image.open(f"results/edits/edit_{i}.jpg").convert("RGB").resize((w, h))
    grid.paste(inp, (0, i * h))
    grid.paste(out, (w, i * h))

grid.save("results/grids/baseline_input_vs_edit.jpg")
print("Saved: results/grids/baseline_input_vs_edit.jpg")