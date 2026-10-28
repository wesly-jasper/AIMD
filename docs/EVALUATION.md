# Evaluating AIMD's detectors

Every change to a model, a threshold or a decision rule is judged the same way: run it on the same held-out split as the current baseline, and compare. A higher recall that comes with too many genuine files flagged is a regression, not an improvement.

The tool is `backend/scripts/evaluate.py`. The code it uses is in `app/services/evaluation/` (datasets, runner, metrics) and `app/services/detection/calibration.py`.

## 1. Dataset

A manifest is a JSON list (or `.jsonl`) next to the files:

```json
{"file": "cam_03_splice.jpg", "label": "manipulated", "group": "cam_03",
 "kind": "splice", "mask": "cam_03.png", "tags": ["jpeg_q75"]}
```

| Field | Meaning |
|---|---|
| `label` | `real`, `manipulated` or `ai_generated` |
| `group` | The source image. An edit and its original share it. Items without one are their own group. |
| `kind` | The manipulation kind (`splice`, `copy-move`, `removal`, `inpainting (GLIDE)` …) or the generator |
| `generator` | For AI-generated items. Defaults to `kind`. |
| `mask` | Pixel ground truth (white = edited), in `masks/` or beside the manifest |
| `split` | `train`, `val`, `test` or `holdout`. Assign it with `split` (below). `holdout` is frozen (§2.1). |
| `tags` | Robustness conditions, such as `jpeg_q75` or `downscaled_0.5`. Metrics are broken down by tag. |

**Which items count for which branch.** The two branches are measured separately and never pooled:

| Label | Manipulation branch | AI-generation branch |
|---|---|---|
| `real` | negative | negative |
| `manipulated` (conventional edit) | positive | **hard negative**: the baseline flagged 30% of edited photos as AI |
| `manipulated` (generative inpainting) | positive | excluded, since it is partly generated |
| `ai_generated` | excluded, since it was not edited after capture | positive |

**What the dataset should contain** (plan Phase 5): verified genuine camera images; images from several generators at several resolutions and compression levels; edited images with pixel masks covering splice, copy-move, object removal and AI inpainting; and hard negatives (heavily compressed, resized, screenshotted or filtered genuine images).

## 2. Split without leakage

```
python scripts/evaluate.py split sets/web/manifest.json --out sets/web/manifest.split.json --val 0.4 --test 0.6
```

- **Splits are by `group`, never by file.** An original and its edit always share a split. Otherwise a threshold fitted on one would be tested on what is nearly the same image.
- **Deterministic.** The same manifest and `--seed` always give the same split.
- **Held-out generators.** `--by-generator` puts each AI generator wholly in one split, to test generalisation to a generator the calibration never saw.
- **Checked.** `validate()` refuses a dataset in which any group appears in two splits, a file or mask is missing, or an id repeats.

### 2.1 The frozen holdout

Every threshold, calibration and detector choice so far was checked on the `test` split, so it no longer gives an unbiased estimate. The **holdout** is a separate test set that nothing is tuned against.

```
python scripts/evaluate.py freeze <set>/manifest.split.json --note "bench2 v1"
```

This writes `backend/evaluation/holdout.lock.json`, which is committed with the code. It lists every holdout file **by content hash** (with id, label, generator and tags), and never the images themselves. The rules are then enforced:

| Command | With a lock present |
|---|---|
| `calibrate` | Refuses `--split holdout`, and refuses any run where a locked file is labelled with another split. |
| `split` | Keeps locked files in the holdout, whatever the new fractions. |
| `run` | Hashes every file. It refuses a holdout file that is not in the lock (changed or added after freezing), and a locked file listed under another split. Holdout runs record the lock in `run.json`. |
| `metrics --split holdout` | States the lock the run was verified against, or warns that it wasn't verified. |
| `freeze` | Refuses to replace a lock unless given `--replace`. Replacing it means results on the old holdout are no longer holdout results, so it is only for a new benchmark version, and the report must say so. |

**How to use it.** Develop on `val` and `test`. Look at the holdout once per release candidate, and report what it says, including when it is worse. If a holdout result prompts a change, the next holdout figure is no longer independent. Collect a new holdout (`--replace`, new version) before claiming a target from it.

### 2.2 The second benchmark (`scripts/build_benchmark.py`)

The first set has 15 GPT-image and 7 Grok/Flux files, no transformed copies, and a test split that has been used for every decision. `build_benchmark.py` builds a second set from Wikimedia Commons to fix all three:

- **Independent sources.**
  - Files are crawled from each generator's Commons categories and their subcategories. A file found in two generators' categories is dropped.
  - Each uploader contributes at most 6 files per generator (3 for real photos). One account's images share a prompt style and pipeline, so they would otherwise count as one example many times over.
  - The uploader is the split `group`, so an account is never in both val and holdout.
  - Anything already in an existing manifest is skipped.
- **Real photographs** come from *Quality images*, sampled at random sort-key prefixes rather than only the category's first titles.
- **Excluded:** titles and categories naming retouches, edits, screenshots, logos, comparisons, collages and diagrams.
- **Transformed copies.** Each original also appears as `jpeg_q70`, `resize_50`, `crop_80`, `webp_q75` and `social` (720 px, JPEG q75, 4:2:0). These are ordinary things that happen to an image after it leaves the generator. Copies share their original's group, so they are always in the same split. Metrics break down by `tag=`.
- **Provenance.** Every item records its Commons page, uploader, licence and SHA-256.

**Adding a set after the holdout is frozen** (bench3):

```
python scripts/build_benchmark.py --out sets/bench3 --seed aimd-bench3-v1 crawl
python scripts/build_benchmark.py --out sets/bench3 --seed aimd-bench3-v1 fetch --per-generator 120 \
    --exclude sets/bench2/manifest.split.json sets/web/manifest.split.json sets/coco/manifest.split.json --exclude-uploaders
python scripts/build_benchmark.py --out sets/bench3 dedupe --against <every other manifest, holdout included>
python scripts/build_benchmark.py --out sets/bench3 --seed aimd-bench3-v1 transform
python scripts/build_benchmark.py --out sets/bench3 --seed aimd-bench3-v1 split --train 0.5 --no-holdout
```

- **`--exclude-uploaders`** skips every uploader already present in an excluded manifest, not only their files. Otherwise a new set would share sources with the frozen holdout.
- **`dedupe`** compares perceptual hashes (pHash, Hamming distance ≤ 8 by default). It drops any original that is a near-duplicate of a file in another set, and puts near-duplicates within the set into one group. It runs before `transform`, and refuses afterwards. Title and uploader checks are not enough: in bench3 it caught five pixel-identical DALL-E re-uploads of older test files under new titles. The results are in `dedupe.json`.
- **`split --no-holdout`** assigns train and val only. The frozen holdout stays the one set for final comparisons.

### 2.3 OpenFake: a differently sourced set, for evaluation only

`scripts/fetch_openfake.py` samples the OpenFake test split (Hugging Face, `ComplexDataLab/OpenFake`) into `sets/openfake`: recent generators (GPT-image 1.5 and 2, FLUX.2, Grok Aurora, Nano Banana Pro, Midjourney 7) and real photos (DOCCI, ImageNet). It reads only the parquet row groups it needs, verifies each image's SHA-256, and keeps files with a short side of at least 256 px.

**Licence: CC BY-NC 4.0.** The images stay local. They are never committed, redistributed, used for training, or used to fit anything that is committed: no calibration, threshold or offset. OpenFake results are reported as a separate collection, never pooled with Commons figures. Commons remains the basis for every committed fit.

## 3. Run

```
python scripts/evaluate.py run sets/web/manifest.split.json sets/coco/manifest.split.json --out runs/<name>
```

**Same code path as production.** Each file goes through the production detectors, with the same timeouts, out-of-memory retry and shared models. The verdicts come from `AnalysisPipeline._decide` and `_decide_ai_generation` themselves, so the runner measures the product rather than a copy of its rule. Video files go through the video engine, including keyframe analysis.

**Output:**
- `run.json`: commit (with `+dirty`), configuration, every detector's version (which names its weights), dataset hash and host.
- `predictions.jsonl`: one line per file, holding:
  - targets and verdicts;
  - each detector's availability and status, raw score, semantics, reason, time and timeout flag;
  - IoU and F1 against the mask, when there is one.

**Resumable:** files already in `predictions.jsonl` are skipped.

## 4. Metrics

```
python scripts/evaluate.py metrics runs/<name> --split test
```

**Per branch:**
- recall, false-positive rate and precision, each with a Wilson 95% interval;
- balanced accuracy;
- score AUC with a bootstrap interval;
- breakdowns by label, kind, generator and tag.

**Abstentions are reported, not hidden.** INCONCLUSIVE and FAILED verdicts don't count as correct or incorrect. Recall and FPR are computed over decided items, and the abstention rate is reported beside them. A detector that abstains on everything has an *unknown* FPR, not a perfect one.

**Also reported:**
- **Localisation:** mean and median IoU, mean F1, and the share of edits with IoU ≥ 0.5. TruFor's map is thresholded at 0.5.
- **Operations:** per-detector status counts, timeouts, and p50/p95 time.
- **Silent failures:** files where an evidential detector FAILED while the verdict still claimed full coverage. The target is 0.

## 5. Compare

```
python scripts/evaluate.py compare runs/baseline runs/candidate --split test
```

- **Per metric:** reports the change, and whether the candidate falls outside the baseline's 95% interval (`improved`, `regressed` or `within_noise`).
- **Exit code:** non-zero when anything regressed, including more silent failures or run errors. That makes it usable as a release gate.

## 6. Calibrate

```
python scripts/evaluate.py calibrate runs/baseline --out models/calibration.json --target-fpr 0.05
CALIBRATION_PATH=models/calibration.json python scripts/evaluate.py run ... --out runs/calibrated
python scripts/evaluate.py compare runs/baseline runs/calibrated --split test
```

**Fitting.** Platt scaling (`p = sigmoid(a·logit(raw) + b)`) is fitted per detector on the **validation** split. The operating threshold is chosen on that split to hold the false-positive rate at or under the target.

**What the record contains:**
- the detector version it was fitted for;
- dataset hash, split and class counts;
- the FPR and recall achieved on the split;
- expected calibration error, before and after.

**Applying it.** With `CALIBRATION_PATH` set, TruFor and UnivFD report `CALIBRATED_PROBABILITY` and decide against the calibrated threshold. This applies only when their version (which includes the weights hash) matches. Otherwise they report the raw score and state why.

**Judge on test, never validation.** The validation figures are optimistic because they were fitted there. Only the test split says whether calibration helped.

**What the probability means.** It is calibrated to the validation set's class balance (`prior` in the record). It answers "how often a file with this score was positive in a set with this mix", not a universal likelihood.

### 6.1 Evidence fusion

```
python scripts/evaluate.py calibrate runs/<val-run> [runs/<another-val-run> ...] --split val --out models/fusion-calibration.json
FUSION_CALIBRATION_PATH=models/fusion-calibration.json   # quantifies evidence; detector flags unchanged
```

Several runs are merged by file id. One run can hold some detectors' results and another run the rest.

**A calibration belongs to one B-Free mode.** B-Free's version names its small-input mode and offset (`.up504o+1.90` for the default). After changing either, rerun B-Free (`--only BFreeDetector`) on the validation sets and fit again. Leave the older runs' B-Free results out of the fit, or `calibrate` skips B-Free for mixing versions. The current fit (v4) and how it was made are in `RELIABILITY_REPORT.md` §4.10.

**Deployed calibrations** are versioned in `backend/calibration/`, with `SHA256SUMS`, the B-Free setting each needs and how to roll back (its `README.md`). `/health` reports whether the configured file is present, matches its checksum and fits every running detector's version.

Each record notes `independent_positives` and `independent_negatives`: distinct originals, with a file's transformed copies counted once. Fusion caps its likelihood ratios on those counts.

**Uncertainty and coverage.** Each calibration also stores:
- **200 bootstrap refits** that resample whole groups, so the copies of one image are one draw. Fusion turns them into a 5th–95th percentile interval per ratio, and grades status and strength on the end nearer 1.
- **Coverage:** the 5th percentile of the fitting images' short side. A file below it gets a direction, not a number.

Runs saved before records kept `image_size` need `--manifest <manifests>`, so that `calibrate` can read the sizes from the files. Without sizes the coverage is recorded as unknown, and no file is excluded.

`calibrate` fits the copy-move detector by **flag rate**, P(fires | edited) against P(fires | genuine), with add-one-half smoothing. Its score is 0 on almost every file, and a sigmoid through that is flat. Every other detector is fitted with Platt scaling.

**Two settings:**
- `FUSION_CALIBRATION_PATH`: the file is used only to turn detector outputs into likelihood ratios for `evidence_fusion`.
- `CALIBRATION_PATH`: also changes detector flags. Fusion falls back to it when `FUSION_CALIBRATION_PATH` is unset.

**How to judge fusion.** Judge on test or holdout with **Cllr** (log-LR cost; below 1 means the ratios carry information) and the rate of ratios of 10× or more pointing the wrong way. Results for the current calibrations are in `RELIABILITY_REPORT.md` §4.5 and §4.8.

## 7. Targets

Proposed targets for release, not current results. See `RELIABILITY_REPORT.md` for the current baseline.

| Metric | Target |
|---|---|
| Manipulation recall | ≥ 85% |
| False-positive rate | ≤ 5% |
| AI-generation balanced accuracy | ≥ 80% |
| Localisation: share of edits with IoU ≥ 0.5 | ≥ 80% |
| Silent detector failures | 0 |

A target is met only when the test-split value meets it with the interval reported beside it.

## 8. Screening a new detector

You don't need to rerun the whole pipeline to compare detectors on one branch:

```
python scripts/evaluate.py run --manifest <m> --out runs/ai-screen --only UnivFDDetector,BFreeDetector
python scripts/evaluate.py metrics --run runs/ai-screen --split test
```

`metrics` prints one line per detector in `BRANCH_DETECTORS`, alongside the branch verdict. A candidate earns a vote only if it beats the incumbent on the test split, with non-overlapping intervals and no rise in false positives. An outvoted detector goes in `AI_GENERATION_ADVISORY_DETECTORS`, where it is still shown but doesn't vote. B-Free against UnivFD is worked through in `RELIABILITY_REPORT.md` §4.3.
