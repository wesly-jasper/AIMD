# AIMD reliability report

**Date:** 2026-10-24 · **Build:** `db4e973` plus uncommitted working-tree changes (see [Environment](#6-environment)) · **Hardware:** NVIDIA RTX 3050 Laptop GPU (4 GB), Windows 11 · **Runtime:** `.venv`, Python 3.11.9, torch 2.14.1+cu126

## 1. Verdict in one paragraph

AIMD is **reliable for one job**: finding and outlining local edits (splices, copy-moves, object removals) in ordinary-resolution photos. TruFor caught 80% of controlled edits at an 8% false-alarm rate, and when it flagged an edit it outlined the right area (mean pixel F1 0.87). Results are fully repeatable, and similarity search never confused two unrelated images.

It is **not reliable** for:
- **AI-generated images:** 10% of 82 were detected.
- **Video manipulation:** 0 of 60 manipulated clips were flagged; scores are at chance.
- **Small or recompressed images:** CocoGlide inpainting detection was 34%, falling to 22% after JPEG q75.
- **Unstable hosts:** under memory pressure the server degrades silently or crashes.

All other image detectors (ELA/FFT/noise, copy-move SIFT, mask generator, face heuristics) abstained on **every** test image. Today the manipulation verdict is effectively TruFor alone.

| Capability | Reliability | Key number (95% CI) |
|---|---|---|
| Local edits in camera photos (960 px) | **Good** | 80% detected [68–88%], 8% false alarms [4–18%], AUC 0.91 |
| Localising the edited area | **Good** when detected | mean pixel F1 0.87; IoU ≥ 0.5 on 85% of edits |
| Generative inpainting, 256 px (CocoGlide) | **Weak** | 34% detected [25–44%], 4% false alarms, AUC 0.76 |
| Same after JPEG q75 | **Poor** | 22% detected [13–35%], AUC 0.66 |
| AI-generated images (6 generators) | **Not usable** | 10% detected [5–18%]; DALL·E 3, Midjourney, Grok 0% |
| Video manipulation (insertion / overlay / deletion) | **Not usable** | 0 / 60 flagged; every clip INCONCLUSIVE; AUC 0.48–0.57 |
| Similarity: copies | **Good** | 50 / 50 re-saved copies verified; 0 false verified matches in 176 |
| Repeatability | **Exact** | 300 / 300 identical verdicts and scores on re-run |
| Robustness to host memory pressure | **Poor** | silent detector failures, a stuck TruFor and a hard crash (§4) |

## 2. Method

**Everything ran through the real API** (`/auth` → `/media/upload` → `POST /analysis` → `GET /analysis/{id}` → TruFor's artefact endpoint).

**Isolated backend:** a separate backend on port 8010 had its own SQLite database and storage. It had no rate or daily limits, so the run neither touched nor was throttled by the dev instance.

**Ground truth is known for every manipulated file:**

| Set | n | What it is | Ground truth |
|---|---|---|---|
| CocoGlide | 100 real + 100 fake | GLIDE generative inpainting of COCO photos, 256 px PNG; each fake is paired with its own original | pixel masks |
| CocoGlide JPEG | 50 + 50 | the first 50 pairs re-saved as JPEG q75, as a messaging app would | pixel masks |
| Camera edits | 60 real + 60 edited | recent "Quality images" from Wikimedia Commons (960 px). 20 each: **splice** (region from another photo), **copy-move** (region from the same photo), **object removal** (OpenCV inpainting). Regions are feathered ellipses covering 4–15% of the image. Real and edited were both saved as JPEG q92, so recompression can't separate them. | pixel masks |
| AI-generated | 82 | Commons categories: DALL·E 2 (15), DALL·E 3 (15), ChatGPT / GPT-image (15), Midjourney (15), Stable Diffusion (15), Grok / Flux (7), as 960 px renditions | label only |
| Video | 20 real + 60 manipulated | real Commons clips (240 p, 8–10 s, 25–30 fps), each re-encoded to MP4 with the same codec, plus 3 manipulations per clip: **insertion** (1.5 s of another clip spliced in), **overlay** (an object pasted for 3 s), **deletion** (a 1.5 s jump cut) | time span, overlay mask |

**Metrics**:
- **Verdict:** SUSPICIOUS / LIKELY_MANIPULATED counts as "flagged", NO_SIGNIFICANT_ANOMALIES_DETECTED as "not flagged". INCONCLUSIVE is reported separately.
- **Confidence intervals:** Wilson 95% intervals for rates; bootstrap intervals for AUC.
- **Localisation:** the localization map is thresholded at 0.5 and compared pixel by pixel with the mask.

Scripts (`fetch.py`, `build_images.py`, `build_videos.py`, `run.py`, `metrics.py`) are in `backend/.scratch/eval/`. Raw results and `metrics.json` are under `%LOCALAPPDATA%\Temp\aimd-reliability\`.

## 3. Results

### 3.1 Image manipulation

| Set | Flagged (recall) | False alarms | Inconclusive | Precision of a flag | TruFor AUC |
|---|---|---|---|---|---|
| Camera edits, all | 48/60 = **80%** [68–88%] | 5/60 = **8%** [4–18%] | 0 | 48/53 = 91% | **0.906** [0.842–0.958] |
| – splice | 17/20 = 85% [64–95%] | | | | 0.951 |
| – object removal | 17/20 = 85% [64–95%] | | | | 0.926 |
| – copy-move | 14/20 = 70% [48–85%] | | | | 0.842 |
| CocoGlide (256 px PNG) | 34/100 = **34%** [25–44%] | 4/100 = **4%** [2–10%] | 0 | 34/38 = 89% | **0.759** [0.691–0.822] |
| CocoGlide, JPEG q75 | 11/50 = **22%** [13–35%] | 2/50 = 4% [1–13%] | 0 | 11/13 = 85% | **0.663** [0.552–0.761] |

**The verdict is TruFor alone.** Across all 502 images analysed (including the AI set), TruFor was the only manipulation detector that ever flagged anything. ForensicImageDetector, CopyMoveDetector, ManipulationMaskGenerator, BaselineFaceDetector and FaceRegionAnalyzer all ran but returned INCONCLUSIVE on every image (0% positives on manipulated and real alike). That includes CopyMoveDetector on the 20 copy-moves. The system's own "corroboration across signal families" rule therefore never fires: every positive was `single_family`, evidence strength WEAK.

**Misses** were edits TruFor scored 0.12–0.45. False alarms were real photos it scored 0.51–0.89, and one of those had 8 marked regions. Copy-moves are the hardest, because the pasted content comes from the same camera and noise.

### 3.2 Localisation (TruFor map vs ground-truth mask)

| Set | Mean pixel F1 | Median F1 | Mean IoU | IoU ≥ 0.5 |
|---|---|---|---|---|
| Camera edits | **0.867** | 0.967 | 0.812 | 51/60 = **85%** [74–92%] |
| – splice / removal / copy-move | 0.965 / 0.835 / 0.802 | | | |
| CocoGlide | 0.335 | 0.230 | 0.260 | 22/100 = 22% [15–31%] |
| CocoGlide, JPEG q75 | 0.194 | 0.002 | 0.148 | 6/50 = 12% [6–24%] |

On camera photos the outline is near-exact. On one splice the true box was (124, 276, 313 × 233) and TruFor's was (127, 275, 310 × 236), pixel IoU 0.99.

### 3.3 AI-generated images (UnivFD, threshold 0.5)

| Generator | Flagged | Median score |
|---|---|---|
| DALL·E 2 | 4/15 = 27% [11–52%] | 0.010 |
| ChatGPT / GPT-image | 2/15 = 13% [4–38%] | 0.007 |
| Stable Diffusion | 2/15 = 13% [4–38%] | 0.171 |
| DALL·E 3 | 0/15 = 0% [0–20%] | 0.002 |
| Midjourney | 0/15 = 0% [0–20%] | 0.010 |
| Grok / Flux | 0/7 = 0% [0–35%] | 0.001 |
| **All** | **8/82 = 10%** [5–18%] | |

- **False "AI" on real images is low:** camera photos 2/60 (3%), CocoGlide originals 0/100, JPEG originals 0/50.
- **The scores barely separate AI from real:** AUC is 0.59 [0.50–0.69] against camera photos.
- **UnivFD reacts to editing, not generation.** It flagged **18 of the 60 edited camera photos (30%)**, against 2/60 of their unedited originals and 8/82 genuinely generated images. So a "Possible AI generation" result on a photo is more often a sign of local editing than of a generated image.
- **Cross-effect on the manipulation verdict:** 9 of the 82 AI images were called SUSPICIOUS (TruFor). 73 were "no significant anomalies".

This matches the detector's documented limit (a linear probe trained on ProGAN), but it confirms that the AI branch can't be relied on for modern generators.

### 3.4 Video

| Clips | Flagged | Verdicts | Suspicious segment over the edit (±1 s) |
|---|---|---|---|
| Real | 0/20 | INCONCLUSIVE ×20 | – |
| Insertion | 0/20 [0–16%] | INCONCLUSIVE ×20 | 0/20 |
| Overlay | 0/20 [0–16%] | INCONCLUSIVE ×20 | 0/20 |
| Deletion | 0/20 [0–16%] | INCONCLUSIVE ×20 | 0/20 |

- **The engine never commits.** The temporal engine (ELA/FFT/noise per frame at 1 fps, optical flow, face heuristic) returned INCONCLUSIVE with rule `no_determination` on every clip.
- **Its score is at chance:** AUC vs real is 0.57 (insertion), 0.50 (overlay) and 0.48 (deletion).
- **No learned model runs on video:** TruFor and UnivFD don't run on frames. The AI-generation branch reports UNAVAILABLE (`image_only`) for video.
- **Coverage gap:** at 1 fps, a 1.5 s insertion covers at most 1–2 sampled frames, and a jump cut is never seen at all.
- **Safe but uninformative:** the output is honest, since it never claims a video is clean, but it gives no information.

### 3.5 Similarity search

| Check | Result |
|---|---|
| Re-saved copy (JPEG q75) verified as the same picture | **50/50** [93–100%] |
| Edited file: its original retrieved | splice 19/20, removal 20/20, copy-move 20/20, CocoGlide 141/150 (94%) |
| VERIFIED_MATCH between unrelated images | **0 of 176** |
| Unrelated images listed as SIMILAR_UNVERIFIED | 128 (a candidate list, not a claim) |
| Edited file VERIFIED as an unedited copy of its original | CocoGlide 41/150, removal 5/20, splice 3/20, copy-move 1/20 |

The last row is the caveat: **VERIFIED_MATCH means "the same picture", not "unedited"**. A small edit (a few % of the area) passes the aligned-pHash gate. Pair a verified match with the manipulation result before calling a file an unaltered copy.

### 3.6 Speed and repeatability

| | Median | p95 | Max |
|---|---|---|---|
| 256 px image | 0.7 s | 0.9 s | 1.6 s |
| 960 px image | 4.5 s | 11.1 s | 52.7 s |
| ≤ 10 s, 240 p video | 1.9 s | 2.7 s | 3.2 s |

- **Model time is small:** TruFor took 0.1–2.3 s and UnivFD under 0.1 s on the GPU.
- **Slow outliers come from elsewhere:** the 25–53 s cases spent their time outside the models, in the CPU heuristic stages or paging while memory was tight.
- **Repeatability:** re-analysing all 300 CocoGlide files gave identical verdicts and identical scores (largest difference 0.0000).

## 4. Operational reliability: what broke during the evaluation

The machine (16 GB RAM) ran out of Windows commit memory twice, because two backends and desktop apps were loaded at once. Each time, the system failed in ways a user couldn't see:

1. **Hard crash while loading the models.**
   - **What happened:** with ~3 GB commit free, the second backend loaded TruFor, then died with `0xC0000005` (access violation) inside `torch.nn.Linear.__init__` while open_clip built CLIP's vision tower.
   - **Expected:** UnivFD marked UNAVAILABLE and the server kept running.
2. **Silent degradation to PARTIAL.**
   - **What happened:** with ~1 GB free, TruFor (`RuntimeError`), CopyMoveDetector and ManipulationMaskGenerator (OpenCV `Insufficient memory`) failed per request.
   - **What the API showed:** the analyses completed as PARTIAL / INCONCLUSIVE. This is honest, but the only clue was a one-word `status_reason`.
3. **TruFor stayed broken after memory was freed.**
   - **What happened:** with 5+ GB free again, every later request still failed with `TruFor inference failed: RuntimeError`, including images that had passed earlier in the same process. UnivFD on the same GPU kept working. Only a restart cleared it.
   - **What `/health` showed:** TruFor still "configured" throughout.
   - **Exception text unknown:** the API keeps only the exception class, and the server log wasn't captured.
   - **Likely cause:** the out-of-memory fallback in `TruForRunner._infer` moves the model to the CPU, retries, then moves it back in a `finally`. If the move back fails, the model is left split across devices and every later call raises.
   - **Fix:** recover or reload the model when that happens, and make `/health` report a detector that is failing on every request.

### 4.1 Fixed since (2026-10-27)

| Failure above | Change | Where |
|---|---|---|
| 1. Hard crash loading CLIP under memory pressure | **Before loading, checks free memory.** TruFor and UnivFD are loaded only when `MODEL_LOAD_MIN_FREE_MB` (default 2048) is free. Otherwise the detector is UNAVAILABLE with the reason stated. **One model copy per process.** `/health` used to build a second detector router of its own, which loaded TruFor a second time; there is now one shared container. | `runtime.memory_preflight`, `detection_container.get_detection_container` |
| 2. Silent degradation, one-word reason | **Timeouts.** Every detector call has a timeout (`DETECTOR_TIMEOUT_SECONDS`, 180 s). A hang becomes FAILED "timed out" instead of a hung request. **One retry after out of memory.** An allocation failure is retried once, after cached memory is released. The reason says "(out of memory)", and the full traceback is logged. | `runtime.run_detector` |
| 3. TruFor stuck after an OOM until restart | **Model recovery.** Each call first checks that the model is wholly on its device. If not, it moves it back, or reloads it from the verified checkpoint. The runner's lock is bounded, so a hung call can't block later ones forever. | `TruForRunner._restore`, `_infer` |
| `/health` said "configured" throughout | **Health from real runs.** `/health` now reports each detector's availability and what its runs since startup did: OPERATIONAL, DEGRADED, FAILING, UNTESTED or UNAVAILABLE. A FAILING detector turns the service `degraded`. **On-demand self-test.** `POST /api/v1/system/detectors/self-test` (admin only) runs every detector on a probe image. `?reload=true` rebuilds the detectors after memory has been freed. | `core/health.py`, `endpoints/system.py` |
| (new) Evidence integrity | **Hash check before analysis.** The pipeline re-hashes the stored file and compares it with the SHA-256 recorded at ingest before any stage runs. A mismatch fails the analysis. | `AnalysisPipeline._verify_integrity` |

### 4.2 Re-measured with the in-process runner (2026-10-27)

This is the same web + CocoGlide data, now split by source image into validation (40%) and test (60%). It ran through `scripts/evaluate.py` (see `EVALUATION.md`). Test split: 331 files.

| | Manipulation | AI generation |
|---|---|---|
| Recall | 60/138 = 43% [35–52%] | 4/55 = 7% [3–17%] |
| False-positive rate | 10/138 = 7% [4–13%] | 15/174 = 9% [5–14%]. Edited photos 13/36 = 36%; real photos 2/138 |
| Balanced accuracy | 0.68 | 0.49 |
| Score AUC | 0.77 [0.71–0.82] | 0.63 [0.54–0.71] |

**Recall by kind:** splice 88%, removal 85%, copy-move 67%, GLIDE inpainting 38%.

**Localisation:** IoU ≥ 0.5 on 48/138 = 35%. This set is dominated by the 256 px CocoGlide images.

**Operations:** 0 run errors and 0 silent failures in 502 files.

**Video, with the new keyframe stage.** All 80 clips ran with 0 errors. The stage detects shot boundaries, selects frames (one per shot plus both sides of every cut), and runs TruFor and UnivFD on each.

| | Real | Insertion | Overlay | Deletion |
|---|---|---|---|---|
| Keyframe TruFor flagged the clip | 3/20 | 4/20 | 5/20 | 1/20 |
| Shot boundary within ±1 s of the edit | – | **20/20** | 1/20 | 1/20 |

- **Cuts localise insertions,** but genuine clips also had 15 cuts between them. Cuts are therefore reported for review, not as evidence.
- **Keyframe TruFor does not separate** manipulated from genuine 240p clips: it flagged 15% of genuine clips and 17% of manipulated ones. Its flags therefore **do not set the video verdict** (`VIDEO_KEYFRAME_VERDICTS_ENABLED=false`). They are shown with timestamps for review.
- **Thread leak found and fixed.** The video run exposed a leak in the detector timeout wrapper. It created a thread per call, and each new thread got its own native OpenMP pool, so the process reached 6,400 threads and 5.6 GB. Workers are now reused per detector.

**Calibration.** Platt scaling was fitted on validation with a 5% FPR target, then applied on test. Every change was **within noise**: manipulation recall 43% → 49%, FPR 7% → 9%, AUC unchanged. That is expected, because a monotone rescaling cannot separate the classes better. Its value is an honest probability and a recorded threshold. **Calibration stays off by default.** The 5% target held on validation but came out at 8.7% on test, because 72 validation negatives are too few to set a threshold reliably.

### 4.3 A second AI-generation detector: B-Free (2026-10-27)

UnivFD missed 93% of generated images in §4.2, so Phase 2 screened a replacement on the same test split. The aim was a gain in recall without more genuine images flagged.

**Candidates:**
- **B-Free** (Guillaro et al., CVPR 2025). A DINOv2 ViT-B/14 scoring five 504 px crops, trained on content-matched real/fake pairs, which is meant to stop it learning scene semantics or JPEG artefacts.
  - Weights: `BFREE_dino2reg4.zip`, MD5 checked, SHA-256 pinned in `bfree_release.py`.
  - Reimplemented in `bfree_runner.py`. A unit test reproduces the authors' reference logits to 1e-3.
- **ClipBased** (Cozzolino et al., CVPR-W 2024). Cloned, not screened. It is the same CLIP-feature idea as UnivFD, which had just failed here.
- **AIDE** (Yan et al., ICLR 2025). Not screened. It needs a ConvNeXt-XXL CLIP backbone, which won't fit beside TruFor in 4 GB of GPU memory.

**Result (test split, 331 files, raw threshold logit > 0, nothing fitted on these files):**

| | UnivFD | **B-Free** |
|---|---|---|
| Recall | 4/55 = 7% [3–17%] | **38/55 = 69% [56–80%]** |
| False-positive rate, all non-generated | 15/174 = 9% [5–14%] | **6/174 = 3% [2–7%]** |
| Of which edited photos | 13/36 = 36% | 2/36 = 6% |
| Of which real photos | 2/138 = 1% | 4/138 = 3% [1–7%] |
| Balanced accuracy | 0.49 | **0.83** |
| Score AUC | 0.63 [0.54–0.71] | **0.86 [0.79–0.92]** |

Validation agrees: recall 74%, FPR 8%, AUC 0.92.

**Reproduced from a clean commit.** The full pipeline was rerun on all 502 files: run `phase2-clean`, commit `9a0f8e5` with no uncommitted changes, dataset `0331254dec71`, with `BFREE_ENABLED=true` and UnivFD advisory, all recorded in `run.json`.
- **AI branch on test:** recall 38/55, FPR 6/174, balanced accuracy 0.83. This matches the screening run exactly.
- **Manipulation branch:** unchanged (recall 60/138, FPR 10/138).
- **Operations:** 0 run errors and 0 silent failures.

That run also showed the branch's "score AUC" had still been computed from UnivFD's score. It now uses the detector that decides the branch: B-Free, 0.86 on test.

The intervals for recall and AUC don't overlap, and B-Free's false positives fall, especially on edited photos. **This is a real improvement by the plan's criterion.**

**By generator (recall):**

| Generator | Recall |
|---|---|
| Stable Diffusion | 11/11 |
| Midjourney | 7/8 |
| DALL-E 3 | 8/10 |
| DALL-E 2 | 7/9 |
| ChatGPT / GPT-image | 3/10 |
| Grok / Flux | 2/7 |

The two newest families together come to 5/17 = 29% [13–53%]. **"Not indicated" still doesn't clear an image made by GPT-image or Flux.**

**Branch policy.** Letting both detectors vote turned 105 of the 331 test verdicts into CONFLICTING, almost all from UnivFD's misses, and left only 22 files indicated.
- **UnivFD is now advisory** (`AI_GENERATION_ADVISORY_DETECTORS=["UnivFDDetector"]`). It is still run and shown, marked "does not count", but it votes only when no other AI detector produced a score.
- **The branch verdict now equals B-Free's figures above.**
- **B-Free is enabled in the local `.env`.** Its licence, like TruFor's, is research / non-profit only. Without the weights, the branch falls back to UnivFD as before.

**Still short of target:** balanced accuracy 0.83 meets the ≥ 0.80 target. FPR 3% meets ≤ 5%, but its interval runs to 7%. With 55 generated test files, a larger test set is needed before calling the target met.

### 4.4 Manipulation detection, Phase 3 (2026-10-27)

**Method.** Screening run `p3-screen`: TruFor and the copy-move detector on all 502 files, with each detector's numeric by-products saved (localisation-map peak and mean, reliability, inlier counts). Every rule below was fitted on **validation** for at most a 5% false-positive rate, then applied unchanged to **test** (138 manipulated, 138 real).

**Copy-move detector fixed.** It had never fired, for two reasons:
- **Self-matching.** It matched each SIFT descriptor against the same set with k=2. The nearest neighbour of a keypoint is itself, and that pair was discarded, so the ratio test almost never compared two real candidates.
- **Wrong geometry.** It verified matches with a fundamental matrix, which models two camera views rather than a region pasted within one image.

Version 2 matches against the two nearest *other* keypoints, orients pairs consistently, and fits a similarity transform with RANSAC. It flags at 50 or more inliers: on validation the 5 copy-moves had 85–413 inliers and the 72 genuine photos at most 35. It reports a box for the source and for the copy.

| Test split | Recall | False positives |
|---|---|---|
| TruFor ≥ 0.5 (current rule) | 60/138 = 43% [35–52%] | 10/138 = 7% [4–13%] |
| Copy-move detector alone | 13/138 = 9% | **0/138 = 0% [0–3%]** |
| **TruFor or copy-move** | **64/138 = 46% [38–55%]** | **10/138 = 7% [4–13%]** |

- **Copy-moves:** 10/15 → **14/15**. The detector flagged none of the 210 genuine files on any split, and 9 of them were inconclusive (too few keypoints).
- **Corroboration.** On 9 test copy-moves both families fire, and the pipeline's own rule grades them **LIKELY_MANIPULATED / STRONG**: the first STRONG verdicts in any evaluation. No genuine file received one.

**Rejected, with the numbers:**

| Change | Recall | False positives | Why rejected |
|---|---|---|---|
| Lower TruFor threshold (0.421, fitted on val) | 67/138 = 49% | 12/138 = 9% | Within noise, in the wrong direction for false positives |
| Decide on the localisation-map peak (the E2E report's H4 idea) | 37/138 = 27% | 10/138 = 7% | Worse than the global score at the same false-positive rate |
| Decide on the localisation-map mean | 18/138 = 13% | 6/138 = 4% | |

**Still open.**
- **AI inpainting:** recall on GLIDE inpainting (256 px CocoGlide) stays at 32/102 = 31%. TruFor wasn't trained on diffusion inpainting at that size, and no rule over its outputs recovers it.
- **B-Free and local fills.** B-Free flagged 4 of 20 classical (OpenCV) object removals as AI, against 1/20 splices, 0/20 copy-moves and 0/60 unedited photos. It was trained on real photos with inpainted patches, so it responds to locally filled-in regions. Its limitations now say so. When it fires on a file the manipulation branch also found edited, the AI-generation result carries an `edit_overlap_note`: both findings may describe the same local edit. The branches are still decided separately.
- **AI inpainting is not the AI-generation branch's job.** The AI-generation detectors answer a different question ("was the whole image generated?"), and partly generated images are excluded from that branch by design (`EVALUATION.md` §1). An inpainting-specific localiser is the next candidate.
- **TruFor stability:** the out-of-memory recovery (§4.1) held in every run since, with 0 run errors in 1,004 file-runs.

### 4.5 Evidence fusion, Phase 4 (2026-10-27)

**What it is.** Each analysis now carries an `evidence_fusion` record beside the verdicts. The verdict rules are unchanged. For each branch, manipulation and AI generation, kept separate, the record answers one question: how much more probable are these detector outputs if the answer is yes than if it is no? That is a likelihood ratio (LR). It is reported as a range with a verbal strength, never as a "probability of manipulation", because the prior (how common edited files are among uploads) is unknown.

- **Calibrated detectors only.** A detector gives a number only with a calibration fitted for its exact version. Otherwise it is listed with its direction and no number.
- **Capped by the data.** An LR is capped at what the validation set could show: with n negatives, at most n + 1.
- **No double counting.** Within a signal family only the strongest detector counts.
- **A range across families.** It runs from fully dependent (only the strongest family counts) to independent (log-LRs add). The verbal strength comes from the weaker end, on the ENFSI scale: < 10 limited, 10–100 moderate, 100–1,000 moderately strong.
- **Conflicts and advisory detectors.** Detectors pointing in opposite directions are listed as conflicts. Advisory detectors (UnivFD) are shown but do not count.
- **Copy-move by flag rate.** Copy-move is calibrated by flag rate, since its evidence is whether it fired: LR ≈ 11 when it fires, ≈ 0.93 when it does not.

**Measured.** Calibrations were fitted on **validation** (TruFor 72+/72−, copy-move 69+/68−, B-Free 27+/96−) and scored on **test**. Cllr is the log-LR cost: 0 is perfect, and 1 is a system that always says LR = 1.

| Test split | Cllr | Positives graded SUPPORTS | Negatives graded SUPPORTS | Negatives graded AGAINST | Misleading by ≥ 10× (positives / negatives) |
|---|---|---|---|---|---|
| Manipulation (138+/138−) | 0.81 | 36 (26%) | 4 (3%) | 9 (7%) | 0.7% / 2.9% |
| AI generation (55+/174−) | 0.61 | 24 (44%) | 1 (0.6%) | 10 (6%) | 1.8% / 0.6% |

- **Mostly "limited".** That is the point: 72% of manipulated and 91% of genuine test files get LIMITED instead of a forced yes or no.
- **Never above "moderate".** The validation sets are small, and the caps are log10(73) and log10(97). More validation data, not a better model, is what lifts that ceiling.
- **Every number holds only for files like the validation data.** The card and the report say so.
- **Checked end to end.** The API, the evidence package, the Markdown/HTML report and the result page all carry the record. A copy-move test file read "Supports, LR 47 to 505, moderate" (TruFor 47, copy-move 11). A missed AI image read "Limited" on both branches, where the verdict had to choose.

**Turning it on.** Set `FUSION_CALIBRATION_PATH` to a calibration file (`EVALUATION.md` §6.1). Without it every branch reads NOT_ASSESSED, with directions only.

### 4.6 Second benchmark and the frozen holdout (2026-10-27)

**Set.** `bench2`: 602 originals from Wikimedia Commons (`EVALUATION.md` §2.2):
- **AI images (362).** 60 each from DALL-E, GPT-image, Imagen / Gemini, Midjourney and Stable Diffusion, plus Grok 37, Flux 15 and Ideogram 10.
- **Unedited photos (240).** At most 6 AI images and 3 photos per uploader.
- **Transformed copies.** Every file also appears five more times: JPEG q70, resized to 50%, cropped to 80%, WebP q75, and a "social media" re-encode (720 px, q75).
- **Split by uploader.** Validation 35%, holdout 65%. The holdout (2,352 files: 1,428 AI, 924 real) was **frozen by content hash before any detector ran on bench2** (`backend/evaluation/holdout.lock.json`, `d959ecd455fe`). It was then evaluated once. Nothing below was tuned on it.

**B-Free, current rule (threshold 0.5), one look at the holdout:**

| | Recall | False positives | AUC |
|---|---|---|---|
| Validation (744 AI / 516 real) | 425/744 = 57% [54–61%] | 7/516 = 1% [1–3%] | 0.91 |
| **Holdout (1,428 AI / 924 real)** | **783/1428 = 55% [52–57%]** | **9/924 = 1% [1–2%]** | **0.89** |
| Holdout, at the threshold calibrated on validation (5% FPR target) | 913/1428 = 64% | 24/924 = 2.6% | |

**Per generator, on the holdout** (recall at 0.5, then at the calibrated threshold):

| Generator | Recall at 0.5 | Recall at calibrated threshold |
|---|---|---|
| Stable Diffusion | 84% | 91% |
| DALL-E | 83% | 90% |
| Ideogram | 69% | 85% |
| Midjourney | 68% | 79% |
| GPT-image | 39% | 50% |
| Imagen / Gemini | 31% | 40% |
| Flux | 25% | 30% |
| Grok | 18% | 30% |

B-Free was trained on Stable Diffusion-family fakes. The generators furthest from that family are missed most often, and those are exactly the ones the roadmap asked about: GPT-image, Grok and Flux.

**Per transform, on the holdout:**

| Transform | Recall | False positives |
|---|---|---|
| Original | 59% | 1/154 |
| JPEG q70 | 59% | 0/154 |
| WebP q75 | 60% | 0/154 |
| Crop to 80% | 58% | 0/154 |
| Social re-encode | 50% | 2/154 |
| **Resize to 50%** | **42%** | **6/154 (4%)** |

Recompression and cropping barely matter. Downscaling is the damaging transform: it costs recall and causes most of the false positives.

**UnivFD** scored AUC 0.51 on validation, no better than chance. It stays advisory, and its fitted calibration is nearly flat (LR ≈ 1).

**Evidence fusion on the holdout.** Calibrations were fitted on all validation data (old sets plus bench2 val). The likelihood-ratio cap counts **distinct originals**, not re-encoded copies: 151 AI and 182 real. Cllr is **0.57**.

| | SUPPORTS | LIMITED | AGAINST |
|---|---|---|---|
| AI images (1,428) | 45% (22% moderately strong) | 52% | 4% |
| Real photos (924) | 0.1% | 81% | 19% |

- **Misleading by 10× or more:** 3.6% of AI images and 0.1% of real photos.
- **Where the misleading ratios come from.** The AI images graded AGAINST are mostly Grok (14% of Grok images) and GPT-image (7%). So **an AGAINST on the AI-generation branch is weak evidence of a camera origin**, and the statement on the card and in the report already says it is not that finding.
- **On the old test split** the same calibrations give Cllr 0.65 (0.61 with the val-only fit), with 2.9% of real files misleading by 10× or more. Calibrations carry over imperfectly between collections, which is why every number holds only "for files like the validation data".

**What did not change.** B-Free's verdict threshold stays 0.5: 55% recall at 1% false positives on the holdout. The calibrated threshold (64% at 2.6%) is available through `CALIBRATION_PATH`. Choosing it is a product decision about false positives, not a measurement.

### 4.7 Why downscaling hurts B-Free (2026-10-28)

**Method.** The bench2 **validation** originals (124 AI, 86 real) were scored again under controlled variants, through the same B-Free preprocessing (five 504 px crops, no resize). Threshold logit 0. The holdout was not used.

| Variant | What it isolates | Recall | False positives | AUC |
|---|---|---|---|---|
| Original | | 58% | 1.2% | 0.926 |
| JPEG q90, full size | recompression alone | 60% | 1.2% | 0.927 |
| Crop to the resized size | fewer pixels, no resampling | 59% | 2.3% | 0.920 |
| Half size, lossless PNG | resampling, no recompression | 44% | 0% | 0.861 |
| Resize to 50% (bench2 transform) | | 44% | 2.3% | 0.846 |
| Resize to 50%, then reflect-padded instead of tiled | B-Free's small-input wrap | 52% | 2.3% | 0.861 |
| Resize to 50%, then upscaled back to 504 px | resolution-aware preprocessing | 44% | 1.2% | 0.912 |

- **Resampling is the cause.** Losing pixels (the crop) and recompression change nothing. Resampling the same picture to half size costs 14 points of recall even when saved losslessly.
- **The traces are smoothed away, and real photos look sharper.** Halving moves AI logits down by about 0.64 and real logits up by about 0.69. Laplacian variance (a sharpness measure) shows why: real photos fall from 2,934 to 2,327, and AI images *rise* from 1,050 to 1,637. Downscaling makes them look alike.
- **Per generator** (original → resized): Stable Diffusion 94% → 61%, Midjourney 86% → 62%.
- **Upscaling small inputs restores the ranking but not the threshold.** Bicubic upscaling to 504 px lifts AUC (0.846 → 0.912), but every logit shifts, so it needs its own offset. Split by group hash: AUC rose on both halves (dev 0.816 → 0.885, check 0.877 → 0.931). With an operating point fitted on dev at 2% false positives, recall on the check half was 72%, against 49% for the tiled default.
- **The offset is not yet stable.** Fitted on all of val it is +0.67; fitted on the dev half, +2.63. Reflect-padding was rejected: its gain was inconsistent between halves, and false positives rose.

**Decision.** Upscale mode is implemented and versioned (`BFREE_SMALL_INPUT_MODE=upscale`, `BFREE_SMALL_INPUT_LOGIT_OFFSET`). The detector version gains `.up504o±x.xx`, so a calibration fitted in one mode is never applied to the other. The default stayed **tile** until the offset was fitted on more Commons reals and confirmed on a differently sourced set. Both are now done, and the default is upscale with offset +2.31 (§4.10). A dedicated low-resolution model was not tested here, because none is installed. SPAI (CVPR 2025) is the obvious candidate; check its licence first.

Every B-Free result now records `image_size`, `input_condition` (`full_size`, `small_tiled` or `small_upscaled`), `raw_logit` and `logit_offset`, so a report says which path a file took.

### 4.8 Calibration reliability: coverage and uncertainty (2026-10-28)

**The question.** On the old test split, Cllr for AI generation rose from 0.57 (holdout) to 0.65. Decomposed with the pool-adjacent-violators bound (Cllr = discrimination + calibration loss):

| Collection, branch | Cllr | Discrimination | Calibration loss |
|---|---|---|---|
| Old test, manipulation | 0.813 | 0.752 | 0.061 |
| Old test, AI generation | 0.647 | 0.528 | 0.119 |
| bench2 val, AI generation (fitted on) | 0.527 | 0.503 | 0.024 |
| bench2 holdout, AI generation | 0.572 | 0.536 | 0.036 |

B-Free discriminates as well on the old test as on the holdout (0.528 against 0.536). **The whole rise is calibration loss**: the ratios were miscalibrated for that collection. Among the old test's real files, the 256 px COCO JPEGs averaged log10 LR −0.16, against −0.46 (camera photos) and −0.52 (COCO PNG). They were misleading by 10× two to four times as often: 5.9% against 1.5–2.8%.

**What changed (calibration file v3).**
- **Uncertainty.** Each calibration carries 200 bootstrap refits that resample whole groups, so the re-encoded copies of one image count as one draw. Each LR gets a 5th–95th percentile interval. **Status and verbal strength now come from the end of that interval nearer 1.** If the interval crosses 1, the result is LIMITED.
- **Coverage.** Each calibration records the 5th percentile of its fitting images' short side. A file below that floor gets the detector's direction only, never a number, and the statement says why.
- **Per-condition calibration was tested and rejected.** Separate B-Free fits for small (< 504 px, tiled) and full-size inputs changed Cllr by less than 0.015 on every collection (old test 0.646 → 0.641; holdout 0.572 → 0.570). The pooled fit stays.

**Do the labels hold?** For each label, the empirical LR is (share of AI images with that label) ÷ (share of real files with it). A "moderate" label claims 10–100.

| Label | v2: old test | v3: old test | v2: holdout | v3: holdout |
|---|---|---|---|---|
| SUPPORTS moderate | **8.7** (11 AI / 4 real) | 41 (26 / 2) | 210 | ∞ (401 / 0) |
| SUPPORTS moderately strong | **73** (23 / 1) | ∞ (7 / 0) | ∞ | ∞ (102 / 0) |
| AGAINST moderate | **0.14** (1 / 22) | 0 (0 / 6) | 0.19 | 0.23 |
| Cllr (cautious end) | 0.647 | 0.624 | 0.572 | 0.597 |
| Misleading by ≥ 10×, AI / real | 1.8% / 2.9% | **0.0% / 1.2%** | 3.6% / 0.1% | **1.1% / 0.0%** |

- **v2 overstated its labels on the old test.** Its "moderate" and "moderately strong" were weaker than claimed. Under v3, every label's empirical LR is within or beyond its stated range on both collections.
- **The price:** more files read LIMITED (holdout AI 52% → 62%), and Cllr on the holdout rises 0.572 → 0.597. That is the intended trade: fewer confident statements, and fewer that point the wrong way.
- **"AGAINST" on the AI branch remains weak evidence.** On the holdout it still came up for 16 AI images, against 46 real ones (empirical 0.23, against a claimed ≤ 0.1). The statement keeps saying it is not a finding that the image came from a camera.

**What it did not fix.** The coverage floor came out at 256 px because the fitting data includes 256 px COCO files, so the COCO JPEGs are inside it. Two of 34 still get "SUPPORTS moderate" under v3 (5.9%, the same rate as v2). Raising the floor would exclude images the calibration was fitted on, and a separate small-input fit did not help. Coverage is about size only. **Small, heavily compressed files are a known weak spot for the AI-generation ratios.**

**Leakage found.** A pHash check of all sets against each other (§4.10) found one picture in the frozen holdout (`bench2/midjourney_046`, 6 files with its copies) identical to a web **val** file the calibrations were fitted on. One bench2 val file is identical to an old **test** file. Both are single images out of hundreds and cannot move the figures above by more than a rounding error. They are recorded rather than removed: the holdout lock is not edited after the fact.

### 4.9 Small AI-inpainted edits (GLIDE) (2026-10-28)

**Recall falls with the edited area** (TruFor at its 0.5 threshold, CocoGlide, 256×256 px):

| Edited area | PNG | JPEG |
|---|---|---|
| 2–5% | 13% (3/23) | 17% |
| 5–10% | 21% (4/19) | 10% |
| 10–20% | 35% (7/20) | 18% |
| > 20% | 53% (20/38) | 35% |

Spearman(area, score) = 0.39. At 256 px, a 5% edit is about 57×57 px.

**Can a different TruFor output separate small edits?** AUC against the authentic COCO images of the same set, for edits under 10% of the image:

| | Integrity score (used now) | Localization-map peak | Map mean |
|---|---|---|---|
| PNG, val / test | 0.64 / 0.63 | 0.64 / 0.67 | 0.54 / 0.57 |
| JPEG, val / test | 0.56 / 0.60 | 0.56 / 0.58 | 0.58 / 0.49 |

All are close to chance. A peak threshold chosen on val, at the integrity flag's val false-positive rate, gave on test 12% recall at 7% false positives (PNG), against 15% at 6% for the current flag. **At an AUC near 0.6, a lower threshold buys recall at almost the same rate in false positives, so TruFor's threshold was not lowered.**

**What would help, not done here.** Small edits need a model that has seen them: a localization model trained or fine-tuned on small inpainted regions, such as CAT-Net v2 or MMFusion (both research-only licences), evaluated on a split of CocoGlide it was not trained on. That requires third-party weights and a licence decision. Until then, a small generative edit is expected to be missed, and "no significant anomalies" must not be read as "not inpainted".

### 4.10 Difficult generators: more independent data (2026-10-28)

**Why.** The holdout has few files from the hardest generators, and every figure so far comes from one source, Wikimedia Commons. Two sets were added: more Commons data (bench3), and a differently sourced set (OpenFake).

**bench3 (Commons, train and val only).** Built with the tools in `EVALUATION.md` §2.2, excluding every uploader already present in another set:
- **Originals:** 115 DALL-E, 64 GPT-image, 27 Midjourney, 10 Stable Diffusion, 7 Flux, 0 Grok, and 98 reals. The uploader exclusion left no Grok files.
- **Dedupe:** the pHash check dropped 6 near-duplicates of older files: five pixel-identical DALL-E re-uploads under new titles, and one GPT-image at distance 8.
- **Results:** B-Free AUC 0.94 on train and 0.96 on val, with 2% false positives on both. GPT-image recall was 41% (train) and 44% (val); Flux 0 of 42. Resize-50 copies lost 21–28 points of recall, as on bench2.
- **The v3 labels held** on bench3 val: "SUPPORTS moderate" had an empirical LR of 53 (claimed 10–100); none of 270 real files got "moderately strong". Cllr 0.42 at the cautious end.

**OpenFake (different source, evaluation only).** 950 AI images from six recent generators, and 450 real photos: 300 from DOCCI and 150 from ImageNet. Its licence (CC BY-NC) rules out fitting anything on it (`EVALUATION.md` §2.3). These figures are not pooled with Commons.

| | Recall | False positives | AUC |
|---|---|---|---|
| All | 33% | 12% (53/450) | 0.59 |
| Nano Banana Pro | 85% | | 0.94 |
| Midjourney 7 | 62% | | 0.85 |
| Grok Aurora | 62% | | 0.84 |
| GPT-image 1.5 / 2 | 22% / 19% | | 0.53 / 0.57 |
| FLUX.2-klein | 6% | | **0.32** |
| DOCCI reals | | **17%** (51/300) | 0.50 against all AI |
| ImageNet reals | | 1% (2/150) | 0.76 against all AI |

AUC per generator is against all 450 reals.

- **B-Free does not generalise to this set as it does on Commons.** The Commons holdout gave 55% recall at 1% false positives; OpenFake gives 33% at 12%.
- **DOCCI is the problem on the real side.** B-Free cannot tell DOCCI photos from AI images (AUC 0.50). They are much smoother than the Commons reals: median Laplacian variance 421, against 2,934. Halving them moves their logits *down* by 2.3, the opposite of what it does to Commons reals (§4.7). That is consistent with the photos having been resampled before release, which §4.7 found makes real photos look AI-generated to B-Free. It is not proven.
- **FLUX.2-klein is ranked below real photos.** This matches Flux at 0 of 42 on bench3.
- **The fusion ratios carry no information here.** Cllr is 1.08 at the cautious end. "SUPPORTS moderate" reached an empirical LR of 9.3, just under its claimed range. The cautious grading limits the damage: 98% of real files read LIMITED, and 1.6% got a ratio misleading by 10× or more.

**Small inputs: upscale mode replaces tiling.** §4.7 left the default at tile until the upscale offset was fitted on more Commons reals and checked on a second source. Only files under 504 px are affected. Across 5,000 full-size files, the scores in the two modes were identical.

**First fit: +2.31, on resized originals.** Fitted on the 50% copies of bench2 val and bench3 train originals (242 AI, 139 real). The offset is the smallest that holds the reals' false-positive rate at tile mode's (5.0%). Probe results on copies under 504 px:

| Set | Tile: recall, FP | Upscale +2.31: recall, FP | AUC |
|---|---|---|---|
| Commons fit | 43%, 5.0% | 65%, 4.3% | 0.85 → 0.91 |
| bench3 val | 48%, 2.2% | 65%, 2.2% | 0.90 → 0.96 |
| OpenFake (402 copies under 504 px) | 36%, 11.7% | 44%, 4.9% | 0.65 → 0.80 |

**+2.31 was too high for compressed files.** Through the full pipeline, real "social" copies on the holdout (720 px, JPEG q75, 4:2:0) went from 2 false positives to 7, out of 154. The fit had seen only half-size copies.

An offset scaled by upscale factor was considered and rejected. The fitting copies were mostly about 480 px (upscaled 1.05×), and AI images lost most there (median logit −2.06, reals −0.68). The offset corrects for downscaling that happened before the file arrived, which the size alone does not reveal.

**Final fit: +1.90, on every small bench2 val input.** That covers resized and social copies and small originals (270 AI, 179 real). Same rule: hold tile mode's false-positive rate (2.8%). Through the full pipeline:

| | Tile | Upscale +1.90 |
|---|---|---|
| Holdout, all: recall / FP | 55% / 9 of 924 | **60% / 10 of 924** |
| Holdout, resize-50 | 42% / 6 of 154 | **53% / 3** |
| Holdout, social | 50% / 2 of 154 | 56% / **5** |
| bench3 val, all | 68% / 6 of 270 | **75% / 6** |
| bench3 val, resize-50 | 48% / 1 of 45 | **61% / 1** |
| OpenFake, all | 33% / 53 of 450 | 33% / 55 |

- **Recall rises on every Commons collection at the same overall false-positive rate.** Holdout AUC 0.887 → 0.891; bench3 val 0.962 → 0.969.
- **Holdout recall by generator** (tile → upscale): Midjourney 68% → 79%, Grok 18% → 27%, GPT-image 39% → 44%, Imagen/Gemini 31% → 35%. Flux is unchanged (25% → 26%).
- **Social copies remain the weak spot:** 5 false positives in 154 against tile's 2.
- **OpenFake barely moves.** Its small files are 154 ImageNet reals and 6 AI images. Two more ImageNet photos were flagged (4 of 154).

**Decision.** The default is now **upscale, offset +1.90** (`BFREE_SMALL_INPUT_MODE`, `BFREE_SMALL_INPUT_LOGIT_OFFSET`). B-Free's version now ends in `.up504o+1.90` for every file. So **a v3 calibration no longer applies to B-Free**: fusion reports its raw score and says why. Both files are in `backend/calibration/`, with checksums and rollback steps. `/health` reports `degraded` when the configured file does not fit the running detectors, and fusion states the reason for each uncalibrated detector.

**Calibration v4.** Refitted on the same validation runs as v3, with B-Free rerun in upscale mode. One file was removed: the web val image identical to a holdout file (§4.8), so v4 no longer has that leak. At 4.9% false positives on val, B-Free recall is 75% (v3: 66%).

| | v3, tile: holdout / old test / bench3 val / OpenFake | v4, upscale: holdout / old test / bench3 val / OpenFake |
|---|---|---|
| Cllr (cautious end) | 0.600 / 0.624 / 0.423 / 1.077 | **0.578 / 0.575 / 0.383** / 1.142 |
| SUPPORTS moderate, empirical LR | ∞ / 41 / 53 / 9.3 | 108 / 51 / 31 / **5.8** |
| SUPPORTS moderately strong, AI files | 101 / 7 / 73 / 15 | **241 / 17 / 141** / 28, no real file on any |
| Misleading by ≥ 10×, AI (cautious) | 1.1% / 0.0% / 0.0% / 0.4% | 1.9% / 1.8% / 0.2% / 2.5% |
| Misleading by ≥ 10×, real (cautious) | 0.0% / 1.2% / 0.8% / 1.6% | 0.2% / 0.6% / 1.1% / 2.7% |

- **On Commons, v4 is better on every collection.** Cllr is lower, and about twice as many AI images get "moderately strong", with no real file. "Moderate" stays within or above its claimed 10–100.
- **The price:** "AGAINST moderate" comes up more often. On the holdout it fell on 27 AI and 91 real files (v3: 16 and 46); its empirical LR is 0.19, against a claimed ≤ 0.1. AGAINST on the AI branch was already weak evidence (§4.8), and it remains so.
- **On OpenFake, v4 is slightly worse, and neither version is usable.** Cllr is above 1 under both, so the ratios carry no information. "SUPPORTS moderate" fell to an empirical 5.8, below its claimed range. "AGAINST moderate" fell on 24 AI images and 1 real one, so on this set it points the wrong way. This is B-Free's failure on OpenFake (above), not something calibration can fix.

### 4.11 End-to-end failure modes and provenance (2026-10-28)

`backend/tests/integration/test_failure_modes_end_to_end.py` drives the real pipeline, API and report generator with a stubbed B-Free runner that misbehaves on demand. It found three defects, all fixed:

| Failure injected | Before | Now |
|---|---|---|
| Score of 1.7 or −0.2 | Accepted as COMPLETED; produced AI_GENERATION_INDICATED | FAILED, "invalid score: not a finite probability in [0, 1]"; the runtime rejects any COMPLETED result whose confidence is outside [0, 1] |
| NaN or infinite logit or score | Reached the JSON response, which then failed to serialize ("Out of range float values are not JSON compliant") | FAILED; non-finite numbers in metadata are replaced by null and listed in `non_finite_values_removed` |
| A malformed result object | "ValidationError" with no detail | "invalid result from the detector" with the fields named |
| GPU out of memory | Retried once (already correct) | Unchanged; tested: two calls, and the reason names memory |
| Timeout | FAILED with a timeout reason (already correct) | Unchanged; tested at 0.5 s |
| Missing weights | UNAVAILABLE, but **UnivFD (AUC 0.51) then decided the AI branch alone** | UNAVAILABLE with the FileNotFoundError; no AI finding |

**Advisory detectors never decide.** Until now, UnivFD voted when B-Free had no score. With B-Free missing, a chance-level detector was making the call. It is now shown for reference only, always, and the branch reads UNAVAILABLE with that reason. Evidence fusion applies the same rule.

**Every result records how it was reached.** The analysis `config_snapshot` and the evidence package now carry:
- **Verdict policy:** a version (`2026-10-28.1`), a SHA-256 of the decision functions' source and the decision settings (advisory list, thresholds, B-Free small-input mode and offset, keyframe verdicts, calibration flags), and `advisory_detectors_vote: false`. Two results with the same hash were decided by the same rules.
- **Calibration files:** path, SHA-256 and the detector versions inside, for both the verdict calibration and the fusion calibration.
- **Detector statuses:** for every detector, its version, availability, execution status, reason, score semantics and calibration summary, including those that failed or never ran.

The Markdown and HTML reports gain a section, "9. Run Provenance", with the software and detector-set versions, the verdict-policy version and hash prefix, both calibration hashes, and the detector status table. The tests check that each of these survives into the package and the report when a detector has failed.

**The test suite itself** had an order-dependent failure: the per-user daily upload quota counted uploads across the whole session's test database, so later files hit HTTP 429. The quota is now lifted for tests that are not about it, the same way the rate limiter already was. Full suite: 747 passed, 2 skipped.

### 4.12 Why B-Free fails on OpenFake (2026-10-28)

**Method.** A sample of each group was scored with the current B-Free setting (upscale, +1.90): 150 DOCCI reals, 80 ImageNet reals, 120 FLUX.2 images and 80 GPT-image images from OpenFake, plus 184 Commons reals (bench2 val and bench3 originals) as the reference. Each original was also scored after six edits that change one thing each. Image properties were measured beside them. Nothing was fitted. Script: `openfake_causes.py`, evaluation data folder.

| | Commons reals | DOCCI | ImageNet | GPT-image | FLUX.2 |
|---|---|---|---|---|---|
| Median logit (flagged) | −5.58 (1%) | **−1.85 (21%)** | −4.39 (0%) | −2.69 (29%) | **−4.96 (7%)** |
| Short side, median | 682 | 1536 | 375 | 1024 | 1024 |
| JPEG quality, chroma | 85, 4:2:0 | **75, 4:2:0** | 95, mostly 4:4:4 | PNG or 90–95 | 95, 4:2:0 |
| Bits per pixel | 2.24 | **1.06** | 5.01 | 3.38 | 3.17 |
| Laplacian variance (sharpness) | 3,428 | **395** | 1,687 | 561 | 1,086 |
| Noise (Immerkaer σ) | 5.41 | **1.04** | 3.80 | 2.07 | 2.57 |

**Median logit change after each edit** (share flagged after):

| Edit | Commons reals | DOCCI | ImageNet | GPT-image | FLUX.2 |
|---|---|---|---|---|---|
| Lossless re-save (check) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Blur, σ 0.8 | +0.05 | **−0.37** (13%) | +0.22 | −0.20 | −0.22 |
| Gaussian noise, σ 2 | +0.01 | +0.06 (24%) | +0.02 | +0.04 | +0.02 |
| Re-encode JPEG q75 | +0.45 | 0.00 (21%) | +0.48 | −0.01 | −0.02 |
| Centre crop to 1024 px | | −0.82 (15%) | | | |
| Resize to 75% | +0.15 | **−1.04** (11%) | −0.01 | −1.43 | −0.88 |

**DOCCI: what it is not.**
- **Not JPEG compression as such.** Re-encoding Commons reals at q75 raises their logits by about 0.45, far short of DOCCI's gap of 3.7. Re-encoding DOCCI changes nothing.
- **Not smoothing.** Blurring DOCCI makes it *less* AI-like, and adding noise changes nothing.
- **Not "having been downscaled" in any way these edits can show.** Downscaling DOCCI further lowers its logits by 1.0. Resampling does not push these photos towards "AI".
- **Not low detail alone.** Commons reals that are as flat and clean as typical DOCCI photos (same Laplacian-variance or noise band, about 24 files) still score −5.3, with none flagged.

**What the data does show.**
- **The flagged DOCCI photos are the flattest ones.** Median Laplacian variance 120 against 1,469 for DOCCI's lowest-scoring quartile; noise 0.48 against 2.61; 0.68 bits per pixel against 1.65. Within DOCCI the logit correlates with these at ρ ≈ −0.6. In every other group, its correlation with any of the three is weaker than 0.35.
- **Less content per crop lowers the score.** B-Free scores five 504 px crops at native resolution. On a 1536 × 2048 photo each crop sees a small, often featureless region. Downscaling puts more of the scene into each crop and lowers DOCCI's logits; cropping the centre (other regions, same scale) also lowers them.
- **The likely explanation is DOCCI's own capture pipeline, which this test did not reproduce.** Clean, high-resolution, low-noise, uniformly processed photos look unlike the camera images B-Free learnt from, and B-Free reads their flat patches as synthetic. This is consistent with everything above, but not proven: no edit made a Commons photo behave like DOCCI.

**FLUX.2 is a recognition failure, not a processing one.** Its images score like real photos (median −4.96), and no edit raises them: blurring, noise and recompression are flat, and downscaling lowers them as for every AI group. B-Free has not learnt FLUX.2's traces. Only a second detector that has learnt them can fix that. None has been evaluated yet.

**Consequence.** A flag on a high-resolution, very clean photo is weaker evidence than the Commons figures suggest, and B-Free's silence on a FLUX.2 image means nothing. Neither can be fixed by preprocessing or calibration. OpenFake stays evaluation only: nothing was fitted on it.

### 4.13 Social-copy false positives: no policy change (2026-10-28)

On the frozen holdout, upscale mode raised false positives on "social" copies (720 px, JPEG q75, 4:2:0) from 2 to 5 of 154 (§4.10). Two remedies were compared **on validation data only**. Each was chosen on bench2 val plus bench3 train (467 AI, 296 real small inputs) and confirmed on bench3 val (182 AI, 92 real):
- **Fallback:** tile mode for a JPEG at quality ≤ Q, upscale otherwise, for Q from 70 to 85.
- **Abstention:** upscale, but INCONCLUSIVE for a compressed file whose logit is within 1 or 2 of the threshold.

| Small inputs | Tile | Upscale (current) | Fallback, Q 75 | Abstain, Q 75, ±1 |
|---|---|---|---|---|
| Choose set: recall / FP | 45% / 11 of 296 | **70% / 12** | 62% / 12 | 65% / 10, 5% abstain |
| Choose set, social copies | 38% / 3 of 91 | **67% / 2** | 38% / 3 | 51% / 1, 16% abstain |
| Confirm set: recall / FP | 47% / 2 of 92 | **68% / 2** | 62% / 2 | 66% / 2, 4% abstain |
| Confirm set, social copies | 45% / 1 of 31 | **66% / 1** | 45% / 1 | 55% / 1, 15% abstain |

- **On validation data, upscale mode does not raise false positives on social copies:** 2 of 91 against tile's 3, and 1 of 31 for both.
- **Fallback costs 6–8 points of recall and removes no false positive.**
- **Abstention removes 2 false positives on the choose set and none on the confirm set.** It costs 4–5% abstentions and 2–5 points of recall.
- **Decision: no change.** The holdout's 2 → 5 does not reproduce on validation data, and is within chance at these counts. The holdout was not inspected again, because no candidate policy was chosen.

## 5. What to rely on, and what not to

**Rely on:**
- A **SUSPICIOUS** verdict with a TruFor region on a normal-resolution photo. About 9 in 10 such flags were real edits in these sets, and the region is accurate.
- **VERIFIED_MATCH** as "same picture". Never between unrelated images here.
- **Determinism:** the same file gives the same answer.
- **Evidence strength graded SUPPORTS or AGAINST** (§4.5), for files like the validation data. A ratio of 10 or more pointed the wrong way for at most 3% of test files.

**Don't rely on:**
- **"No significant anomalies"** as evidence of authenticity. It missed 20% of controlled edits, 66% of generative inpaintings and 78% of recompressed ones, which is what the product wording already says.
- **"AI generation not indicated", or AGAINST on the AI-generation branch.** On the frozen holdout (§4.6), B-Free missed 45% of AI images: 61% of GPT-image, 75% of Flux and 82% of Grok images. With upscale mode (§4.10) it misses 40%, and halving an image's size costs a further 8 points of recall (tile mode: 17).
- **AI-generation verdicts on images unlike the Commons data.** On OpenFake (§4.10), B-Free flagged 17% of DOCCI camera photos and found 6% of FLUX.2 images. Its AUC there was 0.59. A flag on a very clean, high-resolution photo is weaker evidence than the Commons figures suggest, and silence on a FLUX.2 image means nothing (§4.12).
- **"Possible AI generation" on a photo, from UnivFD.** It is more often a sign of editing (§3.3). B-Free flagged 2 of 36 edited photos (§4.3), and UnivFD never votes (§4.11).
- **Any video verdict.** None is ever reached.
- **"Limited" as a finding either way.** It means the detectors barely discriminate on this file.
- **Multi-detector corroboration, except for copy-moves.** Until §4.4 no manipulation detector other than TruFor fired in 502 images. The fixed copy-move detector now corroborates TruFor on clones, and nothing else does yet.
- **VERIFIED_MATCH as proof that a file is unedited** (§3.5).

## 6. Environment

**Two code states.** The CocoGlide sets ran first. Before the web and video sets, the backend had to be restarted (§4, item 3) onto the working tree as changed in the meantime: `db4e973` plus 34 changed files, diff hash `7af99304c078`. That changed four things:
- **Object detector:** `YOLOv8ObjectDetector` replaced the `BaselineObjectDetector` placeholder. It is non-evidential and excluded from verdicts and coverage, and it flagged nothing in these runs.
- **C2PA:** now checked. Every test file returned NO_MANIFEST.
- **Video AI branch:** now reports `image_only`.
- **Live Commons lookup:** added in that code, but **disabled** for the evaluation (`SOURCE_COMMONS_LIVE_ENABLED=false`). The test images come from Commons, so the lookup would trivially find them.

The image decision rule was unchanged between the two states.

**Settings:** TruFor tiles of 1024 px with 128 px overlap (no web image needed more than 2 tiles), UnivFD threshold 0.5, similarity thresholds at their defaults.

## 7. Caveats

- **Synthetic edits:** the camera-photo edits are controlled but simple (feathered ellipses, OpenCV inpainting). Careful human edits and AI inpainting in Photoshop or Firefly are likely harder, as the CocoGlide results suggest. Treat 80% as an upper bound for careful edits.
- **Commons labels are noisy:** some "generated by" files are diagrams, screenshots or composites, and all are 960 px re-encoded renditions, which removes metadata and C2PA. Even so, the low AI recall is consistent across generators.
- **"Real" means unedited by us:** the camera photos are curated "Quality images", which may have been post-processed (levels, sharpening). Some of TruFor's 5 false alarms could be real edits.
- **Small groups:** per-kind and per-generator groups are 15–20 files, hence the wide intervals. The video clips are 240 p, so higher-resolution footage may behave differently.
- **Partial repeatability check:** run-to-run consistency was measured on the CocoGlide set only.
