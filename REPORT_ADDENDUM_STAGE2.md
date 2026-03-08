## Stage 2 Addendum: Experimental Progress and Ongoing Work

### 1) What was added after the baseline

After the initial Stage 2 baseline, we expanded the work from a single img2img setting to a controlled ablation protocol for Prompt 1 ("a smiling person, portrait, high quality").

Concretely, we implemented:

* a dedicated experimental runner (`src/run_prompt1_ablation.py`),
* a scripted sweep over editing parameters (`scripts/run_prompt1_ablation.sh`),
* per-run metric logging and final ranking (`summary_prompt1.json`).

This changed the project from a demo-style baseline into a reproducible experimental setup.

### 2) Experiment design

We evaluate the editability-preservation trade-off on real face inputs under fixed compute settings:

* Model: Stable Diffusion v1.5 (pretrained, no finetuning)
* Data: 20 face images
* Inference steps: 30
* Sweep:
  * `strength`: 0.3, 0.5, 0.6, 0.7
  * `guidance_scale`: 5.0, 7.5, 10.0

Total experiment volume:

* 12 configurations x 20 images = 240 generated edited samples.

### 3) Metrics and current quantitative results

We evaluate each run with:

* `clip_mean` (text-image alignment; higher is better),
* `lpips_mean` (source-preservation drift; lower is better),
* `tradeoff_score = clip_mean - 0.25 * lpips_mean`.

From `results/prompt1_ablation/summary_prompt1.json`:

* `clip_mean` range: 0.2446 to 0.2880
* `lpips_mean` range: 0.2025 to 0.4422
* `tradeoff_score` range: 0.1705 to 0.1945
* Best trade-off configuration: `s0.30_g7.5`
  * `clip_mean = 0.2466`
  * `lpips_mean = 0.2084`
  * `tradeoff_score = 0.1945`

### 4) Interpretation (ready-to-say to instructor)

The ablation confirms the expected behavior of diffusion img2img editing:

* increasing edit intensity (higher `strength`, higher `guidance`) improves prompt alignment,
* but increases perceptual drift from the original image,
* therefore maximizing edit strength alone is not optimal for real-face editing.

This is why we moved to a multi-objective criterion and report the best balance point, not the strongest edit point.

### 5) What is currently in progress

We are extending Stage 2 into a stronger research-grade pipeline:

* add identity-consistency metric (ArcFace cosine),
* add reconstruction/control experiments and failure-case taxonomy,
* compare baseline against inversion/attention-based control in the next stage.

These steps directly address the requirement for deeper theoretical and practical analysis beyond using an off-the-shelf model.


