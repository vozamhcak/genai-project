import argparse
import json
import os
from glob import glob
from typing import Dict, List

import lpips
import numpy as np
import open_clip
import torch
import yaml
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-prompt ablation for editability-preservation trade-off."
    )
    parser.add_argument("--config", type=str, default="configs/multi_prompt.yaml")
    parser.add_argument("--input_dir", type=str, default="data/faces")
    parser.add_argument("--output_dir", type=str, default="results/multi_prompt_ablation")
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


def pil_to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor * 2.0 - 1.0


def pil_to_numpy_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def compute_ssim(inp: Image.Image, out: Image.Image) -> float:
    inp_np = pil_to_numpy_uint8(inp)
    out_np = pil_to_numpy_uint8(out)
    return float(
        ssim(
            inp_np,
            out_np,
            channel_axis=2,
            data_range=255,
        )
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    resolution = int(cfg["resolution"])
    num_steps = int(cfg["num_inference_steps"])
    seed = int(cfg.get("seed", 42))
    num_images = int(args.max_images or cfg["num_images"])

    strengths = [float(x) for x in cfg["strengths"]]
    guidance_scales = [float(x) for x in cfg["guidance_scales"]]
    prompts = cfg["prompts"]

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

    overall_summary: Dict = {
        "model_id": model_id,
        "resolution": resolution,
        "num_inference_steps": num_steps,
        "images_used": len(image_paths),
        "strengths": strengths,
        "guidance_scales": guidance_scales,
        "prompts": [],
    }

    for prompt_spec in prompts:
        prompt_name = prompt_spec["name"]
        edit_prompt = prompt_spec["edit_prompt"]

        prompt_dir = os.path.join(args.output_dir, prompt_name)
        os.makedirs(prompt_dir, exist_ok=True)

        text_tokens = clip_tokenizer([edit_prompt]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        runs: List[Dict] = []

        print(f"\n=== Prompt: {prompt_name} ===")
        print(f"Edit prompt: {edit_prompt}")

        for strength in strengths:
            for guidance in guidance_scales:
                run_name = f"s{strength:.2f}_g{guidance:.1f}"
                run_dir = os.path.join(prompt_dir, run_name)
                os.makedirs(run_dir, exist_ok=True)

                clip_scores: List[float] = []
                lpips_scores: List[float] = []
                ssim_scores: List[float] = []

                print(f"Running {prompt_name} | {run_name} ...")
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

                    ssim_val = compute_ssim(inp, out)
                    ssim_scores.append(float(ssim_val))

                clip_mean = sum(clip_scores) / len(clip_scores)
                lpips_mean = sum(lpips_scores) / len(lpips_scores)
                ssim_mean = sum(ssim_scores) / len(ssim_scores)

                # Main ranking: stronger edit alignment with penalty for drift.
                tradeoff_score = clip_mean - 0.25 * lpips_mean

                run_metrics = {
                    "prompt_name": prompt_name,
                    "edit_prompt": edit_prompt,
                    "run_name": run_name,
                    "strength": strength,
                    "guidance_scale": guidance,
                    "n_images": len(image_paths),
                    "clip_mean": clip_mean,
                    "lpips_mean": lpips_mean,
                    "ssim_mean": ssim_mean,
                    "tradeoff_score": tradeoff_score,
                    "clip_scores": clip_scores,
                    "lpips_scores": lpips_scores,
                    "ssim_scores": ssim_scores,
                }
                runs.append(run_metrics)

                with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(run_metrics, f, indent=2)

        runs_sorted = sorted(runs, key=lambda x: x["tradeoff_score"], reverse=True)
        prompt_summary = {
            "prompt_name": prompt_name,
            "edit_prompt": edit_prompt,
            "model_id": model_id,
            "num_inference_steps": num_steps,
            "images_used": len(image_paths),
            "strengths": strengths,
            "guidance_scales": guidance_scales,
            "ranking_by_tradeoff": runs_sorted,
            "best_run": runs_sorted[0] if runs_sorted else None,
        }

        summary_path = os.path.join(prompt_dir, f"summary_{prompt_name}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(prompt_summary, f, indent=2)

        best = prompt_summary["best_run"]
        print(
            f"Best run for {prompt_name}: {best['run_name']} | "
            f"CLIP={best['clip_mean']:.4f} | LPIPS={best['lpips_mean']:.4f} | "
            f"SSIM={best['ssim_mean']:.4f} | Tradeoff={best['tradeoff_score']:.4f}"
        )

        overall_summary["prompts"].append(
            {
                "prompt_name": prompt_name,
                "edit_prompt": edit_prompt,
                "summary_path": summary_path,
                "best_run": best,
            }
        )

    overall_path = os.path.join(args.output_dir, "overall_summary.json")
    with open(overall_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    print("\nMulti-prompt ablation complete.")
    print(f"Saved overall summary: {overall_path}")


if __name__ == "__main__":
    main()