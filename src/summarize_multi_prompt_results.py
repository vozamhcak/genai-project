import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize multi-prompt ablation results.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/multi_prompt_ablation",
        help="Root directory with prompt subfolders and run metrics.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save summary files. Default: same as input_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    metric_paths = glob(os.path.join(input_dir, "*", "s*_g*", "metrics.json"))
    if not metric_paths:
        raise RuntimeError(f"No metrics.json files found under {input_dir}")

    rows = []
    for path in metric_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows.append(
            {
                "prompt_name": data["prompt_name"],
                "edit_prompt": data["edit_prompt"],
                "run_name": data["run_name"],
                "strength": data["strength"],
                "guidance_scale": data["guidance_scale"],
                "clip_mean": data["clip_mean"],
                "lpips_mean": data["lpips_mean"],
                "ssim_mean": data["ssim_mean"],
                "tradeoff_score": data["tradeoff_score"],
                "n_images": data["n_images"],
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["prompt_name", "tradeoff_score"], ascending=[True, False]).reset_index(drop=True)

    all_runs_csv = os.path.join(output_dir, "all_runs.csv")
    df.to_csv(all_runs_csv, index=False)

    best_df = df.groupby("prompt_name", as_index=False).first()
    best_csv = os.path.join(output_dir, "best_per_prompt.csv")
    best_df.to_csv(best_csv, index=False)

    overall = {
        "input_dir": input_dir,
        "n_runs": int(len(df)),
        "n_prompts": int(df["prompt_name"].nunique()),
        "best_per_prompt": best_df.to_dict(orient="records"),
    }
    overall_json = os.path.join(output_dir, "overall_summary_table.json")
    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    plt.figure(figsize=(8, 6))
    for prompt_name, group in df.groupby("prompt_name"):
        plt.scatter(group["lpips_mean"], group["clip_mean"], label=prompt_name)
        for _, row in group.iterrows():
            plt.annotate(
                row["run_name"],
                (row["lpips_mean"], row["clip_mean"]),
                fontsize=7,
                alpha=0.8,
            )

    plt.xlabel("LPIPS mean (lower = better preservation)")
    plt.ylabel("CLIP mean (higher = better prompt alignment)")
    plt.title("CLIP vs LPIPS across prompts and operating points")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "clip_vs_lpips.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("Saved:")
    print(f" - {all_runs_csv}")
    print(f" - {best_csv}")
    print(f" - {overall_json}")
    print(f" - {plot_path}")


if __name__ == "__main__":
    main()