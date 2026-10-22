# AIMD Sentinel: Zero-Cost Hackathon Plan

Target: **Nebius x NVIDIA Global AI Hackathon 2026**, track **Best Apps and Agents**
Deadline: **30 Oct 2026, 10:00 PT** (17:00 UTC, 22:30 IST). From today (22 Oct) that leaves **8 working days**.
Inputs: `docs/PROJECT_AUDIT.md` (open defects) and the project spec.

---

## 1. What the rules require, and how we meet each one at zero cost

| Rule (Devpost) | How we satisfy it | Cost |
|---|---|---|
| Runs on Nebius Token Factory or Nebius AI Cloud | The Nemotron agent calls Token Factory (OpenAI-compatible API) | Free: $25 hackathon promo credits (code on the Devpost Resources page) + $25 Builders Program credits |
| Uses at least one NVIDIA open model | `nvidia/nemotron-3-super-120b-a12b` (tool calling); fallback `Nemotron-3-Nano-30B-A3B` | At ~$0.06–$0.90 per 1M tokens, one investigation costs < $0.01, so $50 covers thousands |
| A working demo URL | Hugging Face Spaces, free CPU tier, as a Docker Space (2 vCPU / 16 GB RAM) serving the FastAPI app plus the built React app | Free |
| Public repo with an OSS license and setup steps | GitHub repo, MIT `LICENSE`, README with run steps | Free |
| Demo video under 3 minutes | OBS Studio screen recording, uploaded to YouTube (unlisted) | Free |
| A pre-existing project must be significantly updated during the hackathon | The first commit is 2026-08-27, after the period opened on 2026-08-26. Every change is in-period, but the submission text still lists what was built | n/a |
| Explain the NVIDIA / Nebius usage | A section in the README and the Devpost text: model ID, tool list, and a Token Factory request log excerpt | n/a |

**Nothing in the plan needs a paid service.** Every key the code uses is optional. Without one, the dependent feature reports `UNAVAILABLE` and the rest of the pipeline still runs.

---

## 2. Technology choices (all free)

| Capability | Choice | Why | Licence |
|---|---|---|---|
| Branch A: AI-generation detection | **UnivFD** (CLIP ViT-L/14 frozen backbone + linear head, `fc_weights.pth`) | Its weights are small and public, and the CLIP backbone fits in 4 GB at fp16. AIDE needs a larger setup and is a stretch goal | MIT (code + head); CLIP weights are MIT |
| Branch B: manipulation localization | **TruFor** (localization map, reliability map, integrity score) | A real pixel-level localization model, which the spec names | **Research / non-profit only.** Do not vendor it into the repo. `scripts/fetch_models.py` downloads it from GRIP-UNINA at setup time (see risk R1) |
| Supplementary heuristics (ELA, noise, etc.) | Keep, but **demote**. They show as uncalibrated OBSERVATIONs and never drive a verdict | The audit found that they flag every image | ours |
| Similarity search | **Reuse the same CLIP ViT-L/14 embeddings** + pHash/dHash (`imagehash`) + numpy cosine (switch to `faiss-cpu` only if the corpus grows past about 50k) | One model serves two features, which saves RAM on the free host | MIT / BSD |
| Source discovery | (1) **Wikimedia Commons API**: build a real reference corpus with page URLs and upload timestamps. (2) **Wayback Machine CDX API**: earliest-capture timestamps for any URL. (3) **URL verification**: the user pastes a URL, we fetch it, extract `og:image`/`<img>`, and compare by hash and embedding. (4) Optional **Tavily** web/image search with Builders Program credits | Free and keyless (Tavily has a free tier). We do not use Google Vision or TinEye: both are paid, and Bing Visual Search is retired | API ToS-compliant |
| Content credentials | **`c2pa-python`** | Free; reports "no manifest" honestly when one is absent | MIT/Apache |
| Agent | `openai` Python client → `https://api.tokenfactory.nebius.com/v1/`, with native tool calling over 8 allowlisted tools | OpenAI-compatible, so no SDK lock-in | n/a |
| Background jobs | FastAPI `BackgroundTasks` + status polling | No Redis or Celery needed | n/a |
| DB / storage | SQLite + local disk (ephemeral on Spaces, which is acceptable for a demo) | Already in place | n/a |
| Runtime | **Python 3.11** everywhere: create a new venv (the current one is 3.14, which torch/mediapipe/c2pa wheels lag behind) and use the same version in the Dockerfile | Reproducible | n/a |
| Video | Existing keyframe pipeline running both image branches on up to 8 keyframes | Bounded CPU time on the free host | n/a |
| Audio / face-swap classifier | **Out of scope**. They return `UNAVAILABLE` with a stated reason | No validated free model can be integrated in 8 days | n/a |

**Hardware split:** develop and run the evaluation on the RTX 3050 (CUDA, fp16). The hosted demo runs on CPU: TruFor takes roughly 15–40 s per image there, UnivFD roughly 3–6 s. This is acceptable because analysis is a background job and the UI shows progress.

---

## 3. Day-by-day plan

Each day ends with all tests passing and a commit. If a day slips, cut from the **stretch** column, never from the end-to-end path.

### Day 1 (Thu 22 Oct): environment and truthfulness fixes (audit P0)
- Create a Python 3.11 venv and pin `requirements.txt`: add `openai`, `open-clip-torch` or `transformers`, `imagehash`, `c2pa-python`, `mediapipe`; remove unused packages. Add an MIT `LICENSE` and `.env.example`.
- Fix the P0 defects listed in the audit:
  - mask min-max normalisation and the 7-vs-6 weight zip (`manipulation_mask_generator.py:188-196, 366`);
  - the SKIPPED object detector counted as a failure (`analysis_pipeline.py:389-396`);
  - video failure counters that never reset, and failed frames treated as clean (`video_detection_engine.py:83-84, 186-203, 342-354`);
  - the report generated before the final assessment (`analysis_pipeline.py:214` vs `218-231`);
  - every detector labelled `INCONCLUSIVE` (`analysis_pipeline.py:618`);
  - the fake case ID and the unconditional TruFor limitation (`evidence_service.py:61, 265-270`);
  - the 422 `Content-Length` bug (`exceptions.py:171-186`).
- **Done when** a clean camera photo is no longer reported SUSPICIOUS, and regression tests cover each fix.

### Day 2 (Fri 23 Oct): Branch B, real TruFor
- `scripts/fetch_models.py` downloads the TruFor code and weights into `models/` (gitignored) and verifies their SHA-256.
- Rewrite `trufor_runner.py` to load the real architecture and return the localization map, the reliability map and the integrity score. `score_semantics`, `detector_version` and `execution_time_ms` are filled in.
- Missing weights → `UNAVAILABLE` (this already works; keep the test).
- Save the heatmap PNG as an evidence artefact.
- **Done when** a CocoGlide sample produces a heatmap that overlaps its ground-truth mask, and a unit test runs on a fixture image (skipped when no weights are present).

### Day 3 (Sat 24 Oct): Branch A (UnivFD) and similarity
- Add an `ai_generation` detector: UnivFD with fp16 on CUDA and fp32 on CPU, returning a score, semantics ("linear-probe logit on CLIP features; trained on ProGAN; weaker on diffusion models"), and limitations.
- Report the two branches **independently** in the API response and the assessment. Never blend their scores.
- Similarity: compute and store the CLIP embedding + pHash at ingestion; `search_media_fingerprints` returns ranked candidates; a **verification step** (pHash Hamming distance ≤ threshold AND cosine ≥ threshold, plus an ORB/SIFT inlier check) marks a candidate `VERIFIED_MATCH`, `SIMILAR_UNVERIFIED` or `NO_MATCH`.
- **Done when** a resized or re-compressed copy of an indexed image is retrieved and verified, while an unrelated but similar-looking image is not.

### Day 4 (Sun 25 Oct): source discovery, provenance and C2PA
- `scripts/build_reference_corpus.py` pulls about 500–2,000 images from chosen Wikimedia Commons categories (news photos, landmarks), with file URL, page URL, uploader and upload timestamp, and indexes them.
- Add a source connector interface:
  - `WikimediaCorpusSource` (local index);
  - `UrlVerificationSource` (fetch the user's URL with a size and time cap, SSRF guard that blocks private IPs);
  - `WaybackSource` (CDX earliest/latest capture);
  - optional `TavilySource`.
  - Each connector returns `UNAVAILABLE` when disabled.
- Provenance timeline:
  - events typed `PUBLISHED` (upload timestamp), `ARCHIVED` (Wayback), `RETRIEVED` (our fetch time) and `METADATA` (EXIF DateTimeOriginal);
  - every event cites its source;
  - the earliest event is labelled "earliest known appearance", never "original".
- Run C2PA verification on each upload.
- **Done when** uploading a cropped copy of a Commons image yields a verified external source with a real URL and a timeline of at least 2 cited events.

### Day 5 (Mon 26 Oct): the Nemotron investigation agent
- `services/agent/nemotron_agent.py`:
  - tool-calling loop with `max_steps=8`, a 60 s timeout, 2 retries, and logging of latency and token usage;
  - 8 allowlisted tools from the spec, each with a strict JSON schema; arguments are validated with Pydantic, and tools are scoped to the caller's own media and analyses (ownership checks reused);
  - no shell, filesystem or URL arguments beyond `verify_source_match(url)`, which goes through the SSRF guard.
- Grounding guard:
  - the system prompt requires every claim to cite a tool-call ID;
  - after generation, the server checks that every URL, number and timestamp in the summary appears in a tool result, and strips and flags any that do not;
  - the summary is stored labelled `INFERENCE`, with the tool trace stored as `FACT`/`OBSERVATION`.
- Endpoints: `POST /analyses/{id}/investigate` (background) and `GET .../investigation`. A model failure stores `UNAVAILABLE` and never touches existing results.
- Rate-limit the agent per user (for example 10 per hour) so public demo traffic cannot drain the credits.
- **Done when** a real Token Factory request calls at least 2 AIMD tools and the summary's citations resolve. Covered by a recorded-response test plus one live smoke test gated on `NEBIUS_API_KEY`.

### Day 6 (Tue 27 Oct): frontend completion
- The result page shows:
  - two branch cards (AI generation / manipulation) with status, score semantics and limitations;
  - the heatmap overlay on the original with an **opacity slider** and a before/after toggle;
  - the video keyframe timeline;
  - the sources gallery with verification badges;
  - the provenance timeline;
  - an **Investigation** panel for running the agent and viewing the plan, tool trace and cited summary;
  - the report viewer with JSON and HTML/PDF download.
- Every element reads from the real API. Loading, error and `UNAVAILABLE` states are shown explicitly.
- **Done when** the full workflow (upload → analyse → investigate → report) works through the UI only.

### Day 7 (Wed 28 Oct): deploy and evaluate
- Docker image:
  - multi-stage build that runs `npm run build` and lets FastAPI serve `web/dist` (fixes the audit's "nothing serves it");
  - port 7860;
  - `scripts/fetch_models.py` runs at build time;
  - fixes the `docker-compose` volume masking and the missing `AIMD_JWT_SECRET`.
- Push to a **Hugging Face Docker Space**: set `NEBIUS_API_KEY` and `AIMD_JWT_SECRET` as Space secrets, pre-load the reference corpus, and create a demo account.
- `scripts/evaluate.py` on the RTX 3050:
  - **Localization:** CocoGlide (512 images with masks) → pixel F1 / AUC for TruFor.
  - **AI generation:** about 200 real + 200 generated images from a public labelled dataset → precision, recall and FPR, reported **by generator family** (GAN vs diffusion), because UnivFD is weaker on diffusion models.
  - **Robustness:** the same sets after JPEG q=75 and a 50 % resize.
  - **Retrieval:** 100 Commons images × 5 transforms (crop, resize, JPEG, flip, colour shift) → precision@1 and @5; 100 unrelated-but-similar distractors → false-match rate.
  - **System:** end-to-end latency, Token Factory latency, failure rate and report success rate.
  - Write all results to `docs/evaluation.md` with sample sizes. No accuracy claim goes beyond what was measured.

### Day 8 (Thu 29 Oct): documentation, video, submission (with a buffer to the 30 Oct deadline)
- README: what it does, an architecture diagram, setup (`fetch_models`, `.env`, run), how Nemotron and Token Factory are used, and the evaluation summary.
- Also write `docs/limitations.md` and `docs/demo-script.md`.
- Record the demo video (≤ 2:50), on a pre-tested image set:
  1. upload a spliced image → heatmap;
  2. upload an AI-generated image → Branch A flags it, Branch B is clean (shows the branches are independent);
  3. upload a cropped Commons photo → verified source + timeline;
  4. run the Nemotron investigation → tool trace + cited summary;
  5. download the report.
- Fill in the Devpost form: demo URL, repo URL, video, a "built during the hackathon" statement and the NVIDIA/Nebius usage section. **Submit, not just save as draft**, by 29 Oct evening IST.
- Keep 30 Oct as an emergency buffer only.

---

## 4. Scope by priority

| Must ship (submission fails without these) | Should ship | Stretch (only if ahead of schedule) |
|---|---|---|
| P0 truthfulness fixes | Video keyframes through both branches | AIDE as a second Branch A detector |
| TruFor localization + overlay | Wayback timeline events | Nemotron VL describing visible regions (labelled INFERENCE) |
| UnivFD Branch A | C2PA verification | Tavily image search connector |
| Nemotron agent with ≥ 2 real tool calls | Evaluation report | Face-swap classifier |
| Fingerprint match against the Commons corpus | Audit-trail viewer | Audio spoof detection |
| Evidence report (JSON + readable) | | PostgreSQL |
| UI end to end, hosted demo, README, video, LICENSE | | |

---

## 5. What cannot be free, and the honest substitute

| Spec item | Why it is not free | What we ship instead, and how it is labelled |
|---|---|---|
| Search of the open web for where an image has appeared (reverse image search) | Google Vision Web Detection and TinEye are paid, Bing Visual Search is retired, and scraping Google Lens violates its ToS | A verified match against a **real, cited Wikimedia Commons corpus**, plus a **check-this-URL** verifier and the Wayback archive history. The UI says plainly: "searched: Commons corpus (N images), user-provided URLs; not searched: the open web". The existing Google/TinEye provider classes stay as optional, key-gated connectors |
| GPU hosting for the public demo | Nebius AI Cloud GPU time and HF GPU Spaces cost money | A CPU Space: slower (≈ 30–60 s per image) but real. The video is recorded on the local GPU, with that stated |
| Persistent storage on the demo | Free Spaces have ephemeral disk | Acceptable for a demo; state it in the README. The reference corpus is rebuilt at build time |
| TruFor for commercial use | Licence is non-profit or research only | Allowed for a research or demo submission. **We do not redistribute it**; it is fetched from the authors at setup. A commercial product would need a licence or a replacement model (noted in `limitations.md`) |

---

## 6. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Judges ask whether the TruFor licence is compatible with an MIT repo | Our code is MIT; TruFor is an external dependency fetched at setup and attributed. If the organisers object, Branch B falls back to `UNAVAILABLE` and the heuristic maps, which are clearly marked uncalibrated |
| R2 | The free CPU Space runs out of RAM with CLIP-L + TruFor loaded together | Load models lazily, run one analysis at a time (queue), use fp32 CLIP-L at about 1.7 GB and TruFor at under 1 GB, which fits in 16 GB. Fallback: CLIP ViT-B/16 for similarity only |
| R3 | Credits run out, or the API key is abused through the public demo | Per-user agent rate limits, a max-token cap per request, and a daily spend counter in the DB that disables the agent with `UNAVAILABLE` when hit |
| R4 | The Space sleeps during judging (1–15 Dec) | Free Spaces wake on visit, with a cold start of about 1–2 min. Note this in the Devpost text; the video is the backup proof |
| R5 | UnivFD misses modern diffusion images | Report the per-generator evaluation honestly; the limitation appears in every Branch A result. AIDE is the stretch upgrade |
| R6 | The 8-day schedule slips | The scope table above. Video and the Wayback timeline are the first cuts; the end-to-end image path is never cut |
| R7 | The promo code or credit amounts differ from what third-party sites report | **Verify on the Devpost Resources page on Day 1.** Even $1 of credit covers the demo at Nemotron prices |

---

## 7. Accounts to set up on Day 1 (all free)
1. Devpost: register for the hackathon and pick the track.
2. Nebius Token Factory: create an account and redeem the promo credits from the Devpost Resources page. Join the **Nebius AI Builder Program** for the additional credits. Create an API key and confirm the exact Nemotron model ID in the catalogue.
3. Hugging Face: create an account and a Docker Space.
4. GitHub: make the repository public before submission.
5. (Optional) Tavily: free API key.
