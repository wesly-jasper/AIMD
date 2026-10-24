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

## 5. What to rely on, and what not to

**Rely on:**
- A **SUSPICIOUS** verdict with a TruFor region on a normal-resolution photo. About 9 in 10 such flags were real edits in these sets, and the region is accurate.
- **VERIFIED_MATCH** as "same picture". Never between unrelated images here.
- **Determinism:** the same file gives the same answer.

**Don't rely on:**
- **"No significant anomalies"** as evidence of authenticity. It missed 20% of controlled edits, 66% of generative inpaintings and 78% of recompressed ones, which is what the product wording already says.
- **"AI generation not indicated".** It missed 90% of AI images.
- **"Possible AI generation" on a photo.** It is more often a sign of editing (§3.3).
- **Any video verdict.** None is ever reached.
- **Multi-detector corroboration.** No manipulation detector other than TruFor fired in 502 images.
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
