import argparse
import json
import math
import os
import shutil
from glob import glob
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare report-ready assets from multi-prompt results.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/multi_prompt_full",
        help="Directory with multi-prompt results.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/faces",
        help="Directory with original input face images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder. Default: <results_dir>/for_report",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="How many best examples to keep per prompt.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def get_font(size: int = 20):
    # Try a few common fonts, otherwise fallback.
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font, fill=(255, 255, 255)):
    x, y = xy
    # simple shadow for readability
    shadow = (0, 0, 0)
    for dx, dy in [(-1, -1), (1, 1), (1, -1), (-1, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)


def make_labeled_tile(image: Image.Image, title: str, subtitle: str = "", width: int = 320) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    new_h = int(h * width / w)
    image = image.resize((width, new_h))

    pad = 10
    title_h = 34
    subtitle_h = 26 if subtitle else 0
    canvas = Image.new("RGB", (width, new_h + pad * 2 + title_h + subtitle_h), (20, 20, 20))
    canvas.paste(image, (0, title_h + subtitle_h + pad * 2))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(20)
    subtitle_font = get_font(15)
    draw_text(draw, (10, 8), title, title_font)
    if subtitle:
        draw_text(draw, (10, 38), subtitle, subtitle_font)

    return canvas


def hstack(images: List[Image.Image], bg=(255, 255, 255), gap: int = 12) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    out = Image.new("RGB", (sum(widths) + gap * (len(images) - 1), max(heights)), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def vstack(images: List[Image.Image], bg=(255, 255, 255), gap: int = 16) -> Image.Image:
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    out = Image.new("RGB", (max(widths), sum(heights) + gap * (len(images) - 1)), bg)
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + gap
    return out


def grid(images: List[Image.Image], ncols: int, bg=(255, 255, 255), gap: int = 12) -> Image.Image:
    if not images:
        raise ValueError("No images for grid.")
    nrows = math.ceil(len(images) / ncols)
    cell_w = max(im.width for im in images)
    cell_h = max(im.height for im in images)
    out_w = ncols * cell_w + gap * (ncols - 1)
    out_h = nrows * cell_h + gap * (nrows - 1)
    out = Image.new("RGB", (out_w, out_h), bg)

    for idx, im in enumerate(images):
        r = idx // ncols
        c = idx % ncols
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        out.paste(im, (x, y))
    return out


def get_input_paths(input_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(input_dir, "*.jpg")))
    if not paths:
        raise RuntimeError(f"No .jpg input images found in {input_dir}")
    return paths


def load_metrics(results_dir: str, prompt_name: str, run_name: str) -> Dict:
    path = os.path.join(results_dir, prompt_name, run_name, "metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return load_json(path)


def score_image(clip_score: float, lpips_score: float) -> float:
    # Same spirit as run-level tradeoff, only per image.
    return float(clip_score - 0.25 * lpips_score)


def select_top_indices(metrics: Dict, top_k: int) -> List[int]:
    items = []
    for i, (c, l) in enumerate(zip(metrics["clip_scores"], metrics["lpips_scores"])):
        items.append((i, score_image(c, l)))
    items.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in items[:top_k]]


def pick_representative_index(metrics: Dict) -> int:
    top = select_top_indices(metrics, top_k=1)
    return top[0] if top else 0


def read_input_image(input_paths: List[str], idx: int) -> Image.Image:
    return Image.open(input_paths[idx]).convert("RGB")


def read_output_image(results_dir: str, prompt_name: str, run_name: str, idx: int) -> Image.Image:
    path = os.path.join(results_dir, prompt_name, run_name, f"edit_{idx:03d}.jpg")
    return Image.open(path).convert("RGB")


def make_best_gallery_for_prompt(
    results_dir: str,
    input_paths: List[str],
    prompt_name: str,
    best_run: str,
    metrics: Dict,
    out_path: str,
    top_k: int,
):
    indices = select_top_indices(metrics, top_k=top_k)
    rows = []

    for idx in indices:
        inp = read_input_image(input_paths, idx)
        out = read_output_image(results_dir, prompt_name, best_run, idx)

        clip_s = metrics["clip_scores"][idx]
        lpips_s = metrics["lpips_scores"][idx]
        ssim_s = metrics["ssim_scores"][idx]

        left = make_labeled_tile(inp, f"{prompt_name} | input #{idx}")
        right = make_labeled_tile(
            out,
            f"{prompt_name} | {best_run}",
            f"CLIP={clip_s:.3f}  LPIPS={lpips_s:.3f}  SSIM={ssim_s:.3f}",
        )
        rows.append(hstack([left, right]))

    collage = vstack(rows)
    collage.save(out_path)


def make_single_best_overview(
    results_dir: str,
    input_paths: List[str],
    best_df: pd.DataFrame,
    out_path: str,
):
    rows = []
    for _, row in best_df.iterrows():
        prompt_name = row["prompt_name"]
        best_run = row["run_name"]
        metrics = load_metrics(results_dir, prompt_name, best_run)
        idx = pick_representative_index(metrics)

        inp = read_input_image(input_paths, idx)
        out = read_output_image(results_dir, prompt_name, best_run, idx)

        left = make_labeled_tile(inp, f"{prompt_name} | input #{idx}")
        right = make_labeled_tile(
            out,
            f"{prompt_name} | {best_run}",
            f"CLIP={row['clip_mean']:.3f}  LPIPS={row['lpips_mean']:.3f}  SSIM={row['ssim_mean']:.3f}",
        )
        rows.append(hstack([left, right]))
    overview = vstack(rows)
    overview.save(out_path)


def make_parameter_sweep_grid(
    results_dir: str,
    input_paths: List[str],
    all_runs_df: pd.DataFrame,
    prompt_name: str,
    out_path: str,
):
    prompt_df = all_runs_df[all_runs_df["prompt_name"] == prompt_name].copy()
    prompt_df = prompt_df.sort_values(["strength", "guidance_scale"])

    best_row = prompt_df.sort_values("tradeoff_score", ascending=False).iloc[0]
    best_metrics = load_metrics(results_dir, prompt_name, best_row["run_name"])
    idx = pick_representative_index(best_metrics)

    tiles = []
    inp = read_input_image(input_paths, idx)
    tiles.append(make_labeled_tile(inp, f"{prompt_name} | input #{idx}", "reference"))

    for _, row in prompt_df.iterrows():
        run_name = row["run_name"]
        out = read_output_image(results_dir, prompt_name, run_name, idx)
        title = run_name
        subtitle = (
            f"C={row['clip_mean']:.3f} "
            f"L={row['lpips_mean']:.3f} "
            f"S={row['ssim_mean']:.3f}"
        )
        tiles.append(make_labeled_tile(out, title, subtitle))

    g = grid(tiles, ncols=3)
    g.save(out_path)


def make_failure_candidates(
    results_dir: str,
    input_paths: List[str],
    best_df: pd.DataFrame,
    out_path: str,
):
    rows = []
    for _, row in best_df.iterrows():
        prompt_name = row["prompt_name"]
        best_run = row["run_name"]
        best_metrics = load_metrics(results_dir, prompt_name, best_run)

        # under-edit candidate = lowest CLIP in best run
        under_idx = int(min(range(len(best_metrics["clip_scores"])), key=lambda i: best_metrics["clip_scores"][i]))

        # aggressive run = max strength, then max guidance
        aggressive_dir = os.path.join(results_dir, prompt_name)
        run_names = sorted(os.listdir(aggressive_dir))
        aggressive_run = sorted(
            run_names,
            key=lambda x: (
                float(x.split("_")[0][1:]),  # strength
                float(x.split("_")[1][1:]),  # guidance
            ),
            reverse=True,
        )[0]
        aggressive_metrics = load_metrics(results_dir, prompt_name, aggressive_run)

        # over-edit = highest LPIPS in aggressive run
        over_idx = int(max(range(len(aggressive_metrics["lpips_scores"])), key=lambda i: aggressive_metrics["lpips_scores"][i]))
        # artifact / identity drift proxy = lowest SSIM in aggressive run
        drift_idx = int(min(range(len(aggressive_metrics["ssim_scores"])), key=lambda i: aggressive_metrics["ssim_scores"][i]))

        triplet_specs = [
            ("under-edit", best_run, under_idx, best_metrics["clip_scores"][under_idx], best_metrics["lpips_scores"][under_idx], best_metrics["ssim_scores"][under_idx]),
            ("over-edit", aggressive_run, over_idx, aggressive_metrics["clip_scores"][over_idx], aggressive_metrics["lpips_scores"][over_idx], aggressive_metrics["ssim_scores"][over_idx]),
            ("low-ssim", aggressive_run, drift_idx, aggressive_metrics["clip_scores"][drift_idx], aggressive_metrics["lpips_scores"][drift_idx], aggressive_metrics["ssim_scores"][drift_idx]),
        ]

        row_tiles = []
        for label, run_name, idx, c, l, s in triplet_specs:
            out = read_output_image(results_dir, prompt_name, run_name, idx)
            tile = make_labeled_tile(
                out,
                f"{prompt_name} | {label} | #{idx}",
                f"{run_name} | C={c:.3f} L={l:.3f} S={s:.3f}",
            )
            row_tiles.append(tile)

        rows.append(hstack(row_tiles))

    canvas = vstack(rows)
    canvas.save(out_path)


def write_report_notes(best_df: pd.DataFrame, output_path: str):
    lines = []
    lines.append("# Report notes\n")
    lines.append("## Best setting per prompt\n")
    for _, row in best_df.iterrows():
        lines.append(
            f"- **{row['prompt_name']}**: {row['run_name']} | "
            f"CLIP={row['clip_mean']:.4f}, LPIPS={row['lpips_mean']:.4f}, "
            f"SSIM={row['ssim_mean']:.4f}, Tradeoff={row['tradeoff_score']:.4f}"
        )

    lines.append("\n## Short interpretation\n")
    lines.append("- Mild settings worked best for smile/bangs if they preserve identity while still applying the edit.")
    lines.append("- Glasses usually needs a stronger intervention than smile or bangs.")
    lines.append("- Higher edit strength tends to improve CLIP alignment but also increases LPIPS drift and lowers SSIM.")
    lines.append("- For a 2-page report, use only one table, one CLIP-vs-LPIPS plot, one best-examples figure, and one failure-cases figure.\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    results_dir = args.results_dir
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(results_dir, "for_report")
    ensure_dir(output_dir)

    input_paths = get_input_paths(input_dir)

    best_csv = os.path.join(results_dir, "best_per_prompt.csv")
    all_runs_csv = os.path.join(results_dir, "all_runs.csv")
    plot_png = os.path.join(results_dir, "clip_vs_lpips.png")
    overall_json = os.path.join(results_dir, "overall_summary_table.json")

    if not os.path.exists(best_csv):
        raise FileNotFoundError(f"Missing {best_csv}")
    if not os.path.exists(all_runs_csv):
        raise FileNotFoundError(f"Missing {all_runs_csv}")

    best_df = pd.read_csv(best_csv)
    all_runs_df = pd.read_csv(all_runs_csv)

    # Copy core artifacts
    copy_if_exists(best_csv, os.path.join(output_dir, "best_per_prompt.csv"))
    copy_if_exists(all_runs_csv, os.path.join(output_dir, "all_runs.csv"))
    copy_if_exists(plot_png, os.path.join(output_dir, "clip_vs_lpips.png"))
    copy_if_exists(overall_json, os.path.join(output_dir, "overall_summary_table.json"))

    # Write concise notes
    write_report_notes(best_df, os.path.join(output_dir, "report_notes.md"))

    # Best overview across prompts
    make_single_best_overview(
        results_dir=results_dir,
        input_paths=input_paths,
        best_df=best_df,
        out_path=os.path.join(output_dir, "best_overview.png"),
    )

    # Per-prompt galleries and sweep grids
    for _, row in best_df.iterrows():
        prompt_name = row["prompt_name"]
        best_run = row["run_name"]
        metrics = load_metrics(results_dir, prompt_name, best_run)

        make_best_gallery_for_prompt(
            results_dir=results_dir,
            input_paths=input_paths,
            prompt_name=prompt_name,
            best_run=best_run,
            metrics=metrics,
            out_path=os.path.join(output_dir, f"best_gallery_{prompt_name}.png"),
            top_k=args.top_k,
        )

        make_parameter_sweep_grid(
            results_dir=results_dir,
            input_paths=input_paths,
            all_runs_df=all_runs_df,
            prompt_name=prompt_name,
            out_path=os.path.join(output_dir, f"sweep_grid_{prompt_name}.png"),
        )

    # Failure candidates
    make_failure_candidates(
        results_dir=results_dir,
        input_paths=input_paths,
        best_df=best_df,
        out_path=os.path.join(output_dir, "failure_candidates.png"),
    )

    print("Prepared report assets in:")
    print(output_dir)
    print("\nCreated files:")
    for path in sorted(glob(os.path.join(output_dir, "*"))):
        print(" -", os.path.basename(path))


if __name__ == "__main__":
    main()