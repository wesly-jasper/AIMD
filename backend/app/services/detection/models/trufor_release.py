"""
The pinned TruFor release this codebase is written against.

TruFor (Guillaro et al., CVPR 2023, GRIP-UNINA) is licensed for research and
non-profit use only, so neither its code nor its weights are in this repository.
``scripts/fetch_models.py`` downloads them at setup time and verifies every file
against the hashes below; ``TruForRunner`` verifies them again before importing
any of it. A file that does not match is never executed.

Constants only, so the fetch script can import this module without pulling in
torch.
"""
from __future__ import annotations

UPSTREAM_REPO = "grip-unina/TruFor"
UPSTREAM_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
UPSTREAM_RAW_BASE = (
    f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/test_docker/"
)
LICENCE_NOTE = (
    "TruFor code and weights: (c) 2023 GRIP-UNINA, research and non-profit use only "
    "(LICENSE.txt). CMX components: MIT (src/models/cmx/LICENSE_CMX.txt)."
)

#: Inference code, keyed by the path below ``test_docker/`` upstream, which is
#: also the path below the local TruFor root. ``config.py`` (needs yacs) and the
#: dataset/test scripts are deliberately absent: the runner builds the model
#: config itself and does its own preprocessing.
CODE_FILES: dict[str, str] = {
    "LICENSE.txt": "07201e07e3d2c1ac55480037a87734fcccacbb0cd0e25a31e3b89ac7ffadf8b4",
    "src/models/__init__.py": "e8bba2be4a95dbbe0e3a0fe6701ebfc72518206b9a6ed3b4f5e349cb8f98a578",
    "src/models/DnCNN.py": "3736b7c625d03b85accea77c26a41ea2da956dae07d70182532e7df9b74ae882",
    "src/models/cmx/LICENSE_CMX.txt": "687ff4d0ea13200541df7359799a08c2db626094ecacbb8a3fe63ffad177f2a1",
    "src/models/cmx/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/models/cmx/builder_np_conf.py": "4c89614d242df222810f2f172886b403eb808289fd75dfa78e59728ae9537eb9",
    "src/models/cmx/decoders/MLPDecoder.py": "745b3fba29f0a3e52cb7e40080ac8bb2c53891399015ff32bbf2e86cc116fbe5",
    "src/models/cmx/encoders/dual_segformer.py": "e34671a3e8bb5e031a36a6ea9739531616ad4b35d9d5367746bdbf49c98520e8",
    "src/models/cmx/layer_utils.py": "f24723cec04bf5b9296deaaa163bdc0dd1e3d7599fe1e2ba1450c4f06a98fc95",
    "src/models/cmx/net_utils.py": "171239507b49f00026ca46db88cbb101f8c24f47dd7a93b1b5a7f919deb8ed21",
    "src/models/cmx/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "src/models/cmx/utils/init_func.py": "2a3a1c7da372079b8592612261cc9865127c292aea15d16eae957545a8585881",
}

#: The upstream demo images, used as test fixtures. Optional.
SAMPLE_FILES: dict[str, str] = {
    "images/tampered2.png": "893e4a9079b82d4d657b1e1b47fa69ab79345c30074637aa8c77b7e604f7f3b2",
    "images/pristine1.jpg": "b09e0a4600f7e7bf0caedb1bc42454b4b7ebbc2d1af7dcdf1ab219c9a33c250c",
}

#: Official weights archive. The authors publish an MD5 only, so that is what an
#: archive is verified against; its SHA-256, and the SHA-256 of the checkpoint
#: inside it, are recorded in the local manifest when it is installed.
WEIGHTS_URL = "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
WEIGHTS_ZIP_MD5 = "7bee48f3476c75616c3c5721ab256ff8"
CHECKPOINT_NAME = "trufor.pth.tar"

#: CocoGlide evaluation set (512 images with ground-truth masks), same host.
COCOGLIDE_URL = "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip"

#: Layout below the local TruFor root (``backend/models/trufor`` by default).
MANIFEST_NAME = "manifest.json"
WEIGHTS_RELPATH = f"weights/{CHECKPOINT_NAME}"
SRC_RELPATH = "src"
SAMPLES_RELPATH = "samples"
COCOGLIDE_RELPATH = "CocoGlide"
