# Calibration files

Versioned calibrations for evidence fusion (`FUSION_CALIBRATION_PATH`). Each entry applies only to the exact detector version it was fitted for. A mismatched detector is reported as direction only, never given a number. `/health` compares every configured file with the detectors actually running, and reads `degraded` on a mismatch, a missing or unreadable file, or a checksum that does not match `SHA256SUMS`.

| File | Status | B-Free setting it requires | Fitted on (val split) |
|---|---|---|---|
| `fusion-calibration.v4.json` | **current** | `BFREE_SMALL_INPUT_MODE=upscale`, `BFREE_SMALL_INPUT_LOGIT_OFFSET=1.90` (the defaults) | web + coco + bench2 val; B-Free rerun in upscale mode; the web file duplicating a holdout image removed |
| `fusion-calibration.v3.json` | rollback | `BFREE_SMALL_INPUT_MODE=tile`, `BFREE_SMALL_INPUT_LOGIT_OFFSET=0` | web + coco + bench2 val, tile mode |

Measurements behind both are in `docs/RELIABILITY_REPORT.md`: §4.8 for v3, §4.10 for v4.

## Deploy

```
FUSION_CALIBRATION_PATH=calibration/fusion-calibration.v4.json   # local, relative to backend/
```

The Docker image also copies this folder to `/app/calibration`, and `docker-compose.yml` points there. Compose mounts a volume over `/app/backend`, and that volume would keep serving the files of the first image ever run.

Then check `/health`. Under `detectors.calibration.fusion_calibration_path`:
- `state` must be `ok` and `checksum` must be `verified`;
- every running detector must be `matched`, B-Free included;
- `consistent` must be `true`.

## Roll back to v3

```
BFREE_SMALL_INPUT_MODE=tile
BFREE_SMALL_INPUT_LOGIT_OFFSET=0
FUSION_CALIBRATION_PATH=calibration/fusion-calibration.v3.json
```

Changing only one of these leaves B-Free uncalibrated, and `/health` reports `version_mismatch`.

## Adding a file

Fit it with `scripts/evaluate.py calibrate` (`docs/EVALUATION.md` §6), never on the frozen holdout or on OpenFake. Copy it here under the next version number, add its line to `SHA256SUMS` (`sha256sum fusion-calibration.vN.json >> SHA256SUMS`), and add a row above.

**Contents.** Each file holds fitted parameters, bootstrap refits, class counts, the evaluation runs' names and their datasets' hashes. It holds no image data.
