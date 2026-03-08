import argparse
import json
import os
from glob import glob
from typing import Dict, List, Tuple

import lpips
import open_clip
import torch
import yaml
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prompt 1 (smile) ablation for editability-preservation trade-off."
    )
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--input_dir", type=str, default="data/faces")
    parser.add_argument("--output_dir", type=str, default="results/prompt1_ablation")
    parser.add_argument(
        "--strengths",
        type=str,
        default="0.3,0.5,0.6,0.7",
        help="Comma-separated list. Example: 0.3,0.5,0.6,0.7",
    )
    parser.add_argument(
        "--guidance_scales",
        type=str,
        default="5.0,7.5,10.0",
        help="Comma-separated list. Example: 5.0,7.5,10.0",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Override num_images from config if provided.",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def pil_to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    # LPIPS expects tensor in [-1, 1] with shape [N, C, H, W].
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor * 2.0 - 1.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    resolution = int(cfg["resolution"])
    num_steps = int(cfg["num_inference_steps"])
    edit_prompt = cfg["edit_prompt"]
    seed = int(cfg.get("seed", 42))
    num_images = int(args.max_images or cfg["num_images"])

    strengths = parse_float_list(args.strengths)
    guidance_scales = parse_float_list(args.guidance_scales)

    image_paths = sorted(glob(os.path.join(args.input_dir, "*.jpg")))[:num_images]
    if not image_paths:
        raise RuntimeError(
            f"No input .jpg images found in {args.input_dir}. "
            "Add images first (e.g., data/faces/*.jpg)."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Device: {device}, images: {len(image_paths)}")

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    pipe.safety_checker = None

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    text_tokens = clip_tokenizer([edit_prompt]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    runs: List[Dict] = []

    for strength in strengths:
        for guidance in guidance_scales:
            run_name = f"s{strength:.2f}_g{guidance:.1f}"
            run_dir = os.path.join(args.output_dir, run_name)
            os.makedirs(run_dir, exist_ok=True)

            clip_scores: List[float] = []
            lpips_scores: List[float] = []

            print(f"Running {run_name} ...")
            for i, path in enumerate(image_paths):
                inp = Image.open(path).convert("RGB").resize((resolution, resolution))
                generator = torch.Generator(device=device).manual_seed(seed + i)
                out = pipe(
                    prompt=edit_prompt,
                    image=inp,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=num_steps,
                    generator=generator,
                ).images[0]

                out_path = os.path.join(run_dir, f"edit_{i:03d}.jpg")
                out.save(out_path)

                with torch.no_grad():
                    img = clip_preprocess(out).unsqueeze(0).to(device)
                    img_feat = clip_model.encode_image(img)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    clip_sim = (img_feat @ text_feat.T).item()
                    clip_scores.append(float(clip_sim))

                    inp_lp = pil_to_lpips_tensor(inp).to(device)
                    out_lp = pil_to_lpips_tensor(out).to(device)
                    lpips_val = lpips_model(inp_lp, out_lp).item()
                    lpips_scores.append(float(lpips_val))

            clip_mean = sum(clip_scores) / len(clip_scores)
            lpips_mean = sum(lpips_scores) / len(lpips_scores)
            # Higher is better: maximize edit alignment while penalizing drift.
            tradeoff_score = clip_mean - 0.25 * lpips_mean

            run_metrics = {
                "run_name": run_name,
                "strength": strength,
                "guidance_scale": guidance,
                "n_images": len(image_paths),
                "clip_mean": clip_mean,
                "lpips_mean": lpips_mean,
                "tradeoff_score": tradeoff_score,
                "clip_scores": clip_scores,
                "lpips_scores": lpips_scores,
            }
            runs.append(run_metrics)

            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(run_metrics, f, indent=2)

    runs_sorted = sorted(runs, key=lambda x: x["tradeoff_score"], reverse=True)
    summary = {
        "prompt": edit_prompt,
        "model_id": model_id,
        "num_inference_steps": num_steps,
        "images_used": len(image_paths),
        "strengths": strengths,
        "guidance_scales": guidance_scales,
        "ranking_by_tradeoff": runs_sorted,
        "best_run": runs_sorted[0] if runs_sorted else None,
    }

    summary_path = os.path.join(args.output_dir, "summary_prompt1.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best = summary["best_run"]
    print("\nAblation complete.")
    print(f"Saved summary: {summary_path}")
    print(
        f"Best run: {best['run_name']} | clip_mean={best['clip_mean']:.4f} | "
        f"lpips_mean={best['lpips_mean']:.4f} | tradeoff={best['tradeoff_score']:.4f}"
    )


if __name__ == "__main__":
    main()
