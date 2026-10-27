"""
The pinned B-Free model this codebase is written against.

B-Free (Guillaro, Zingarini, Cozzolino, Verdoliva, CVPR 2025, "A Bias-Free
Training Paradigm for More General AI-generated Image Detection") is a ViT-B/14
with four registers, initialised from DINOv2 and trained on fakes generated from
real images by Stable Diffusion's own conditioning, so real and fake share their
content and the detector cannot learn content shortcuts. It outputs one logit;
``logit > 0`` is the authors' "synthetic" decision.

Research / non-profit licence, like TruFor (grip.unina.it LICENSE_OPEN.txt), so
the weights are fetched at setup time by ``scripts/fetch_bfree.py``, never
committed. The archive is checked against the authors' MD5 and the checkpoint
against the SHA-256 below, at install time and again before it is loaded.

Constants only, so the fetch script can import this module without torch.
"""
from __future__ import annotations

UPSTREAM_REPO = "grip-unina/B-Free"
UPSTREAM_COMMIT = "c6a9f89"
LICENCE_NOTE = (
    "B-Free: (c) 2025 GRIP-UNINA, University Federico II of Naples; informational "
    "and non-profit use only (https://www.grip.unina.it/download/LICENSE_OPEN.txt)."
)

ARCHIVE_URL = "https://www.grip.unina.it/download/prog/B-Free/weights/BFREE_dino2reg4.zip"
#: Published by the authors in code/README.md.
ARCHIVE_MD5 = "f3f53fa647848b16cf81c913f148a198"

MODEL_DIR = "BFREE_dino2reg4"
WEIGHTS_FILE = "model_epoch_best.pth"
#: Computed from the checkpoint inside the MD5-verified archive.
WEIGHTS_SHA256 = "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947"

#: timm architecture (config.yaml: ``timm_c5i504_`` + this) and its input.
TIMM_ARCH = "vit_base_patch14_reg4_dinov2"
CROP_SIZE = 504
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MANIFEST_NAME = "manifest.json"
