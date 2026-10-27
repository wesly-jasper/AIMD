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
