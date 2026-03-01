import torch
import open_clip
from PIL import Image
from glob import glob
import json, os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

prompt = "a smiling person, portrait, high quality"
text = tokenizer([prompt]).to(device)

paths = sorted(glob("results/edits/edit_*.jpg"))[:20]

scores = []
with torch.no_grad():
    text_feat = model.encode_text(text)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    for p in paths:
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        img_feat = model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ text_feat.T).item()
        scores.append(sim)

os.makedirs("results", exist_ok=True)
out = {"prompt": prompt, "n": len(scores), "clip_mean": sum(scores)/len(scores), "clip_scores": scores}
with open("results/metrics_clip.json", "w") as f:
    json.dump(out, f, indent=2)

print("CLIP mean:", out["clip_mean"])
print("Saved: results/metrics_clip.json")