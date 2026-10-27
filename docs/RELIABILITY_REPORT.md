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
- **AI inpainting is not the AI-generation branch's job.** The AI-generation detectors answer a different question ("was the whole image generated?"), and partly generated images are excluded from that branch by design (`EVALUATION.md` §1). An inpainting-specific localiser is the next candidate.
- **TruFor stability:** the out-of-memory recovery (§4.1) held in every run since, with 0 run errors in 1,004 file-runs.

## 5. What to rely on, and what not to

**Rely on:**
- A **SUSPICIOUS** verdict with a TruFor region on a normal-resolution photo. About 9 in 10 such flags were real edits in these sets, and the region is accurate.
- **VERIFIED_MATCH** as "same picture". Never between unrelated images here.
- **Determinism:** the same file gives the same answer.

**Don't rely on:**
- **"No significant anomalies"** as evidence of authenticity. It missed 20% of controlled edits, 66% of generative inpaintings and 78% of recompressed ones, which is what the product wording already says.
- **"AI generation not indicated".** UnivFD missed 90% of AI images. B-Free (§4.3) misses 31% overall, and about 70% of GPT-image and Flux images.
- **"Possible AI generation" on a photo, from UnivFD.** It is more often a sign of editing (§3.3). B-Free flagged 2 of 36 edited photos (§4.3), and UnivFD no longer votes when B-Free has a score.
- **Any video verdict.** None is ever reached.
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
