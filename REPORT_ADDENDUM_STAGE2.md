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

### 6) Team contribution wording (ready text)

** Albert (theory + methodology):**
defined the editability-preservation objective, designed ablation factors, and prepared interpretation of trade-off behavior.

** Alexander (implementation + evaluation):**
implemented the sweep runner, metric logging pipeline, and reproducible experiment scripts/artifacts.

**Joint contribution:**
analysis of outputs, selection of best operating point, and preparation of the Stage 2 addendum.

### 7) Ready-to-insert text for previous report

#### 7.1 Stage 2 progress paragraph

After submitting the initial baseline, we expanded Stage 2 from a single img2img setting to a controlled ablation study on Prompt 1. We evaluated 12 parameter configurations (`4 strengths x 3 guidance scales`) on 20 real face images, resulting in 240 generated outputs. For each configuration we logged text-alignment (`clip_mean`), source-preservation drift (`lpips_mean`), and a combined trade-off score. This transformed our work from a demo baseline into a reproducible experimental protocol with measurable operating-point selection.

#### 7.2 Quantitative results paragraph

Our ablation produced the following ranges: `clip_mean` from 0.2446 to 0.2880, `lpips_mean` from 0.2025 to 0.4422, and `tradeoff_score` from 0.1705 to 0.1945. The best trade-off setting was `s0.30_g7.5` with `clip_mean = 0.2466`, `lpips_mean = 0.2084`, and `tradeoff_score = 0.1945`. The results indicate that stronger editing parameters increase prompt alignment, but also increase perceptual drift from the source image.

#### 7.3 Theoretical interpretation paragraph

These results empirically validate the central hypothesis of Stage 2: real-face diffusion editing is a multi-objective optimization problem, where editability and identity/content preservation are in tension. Maximizing edit intensity alone does not yield the best practical configuration. Therefore, we evaluate and report balanced operating points rather than only the strongest visual edits.

#### 7.4 Ongoing work paragraph

We are currently extending the evaluation stack with identity consistency (ArcFace cosine) and structured failure-case taxonomy. In the next stage, we will compare this baseline against inversion- and attention-based control methods to improve locality and identity preservation while maintaining editability.
