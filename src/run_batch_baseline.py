from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Device:", device)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

pipe.safety_checker = None

os.makedirs("results/edits", exist_ok=True)

prompt = "a smiling person, portrait, high quality"

image_paths = glob("data/faces/*.jpg")[:20]

for i, path in enumerate(image_paths):
    image = Image.open(path).convert("RGB").resize((512, 512))

    os.makedirs("results/inputs", exist_ok=True)
    image.save(f"results/inputs/in_{i}.jpg")

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.6,
        guidance_scale=7.5,
        num_inference_steps=30,
    ).images[0]

    out_path = f"results/edits/edit_{i}.jpg"
    result.save(out_path)
    print("Saved:", out_path)
