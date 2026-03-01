from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)
pipe.safety_checker = None

os.makedirs("results/recon", exist_ok=True)

prompt = "a person, portrait, high quality"

inputs = sorted(glob("results/inputs/in_*.jpg"))[:20]

for i, path in enumerate(inputs):
    image = Image.open(path).convert("RGB").resize((512, 512))
    out = pipe(
        prompt=prompt,
        image=image,
        strength=0.2,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]
    out.save(f"results/recon/recon_{i}.jpg")
    print("Saved:", f"results/recon/recon_{i}.jpg")
