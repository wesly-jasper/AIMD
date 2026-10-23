"""
The pinned CLIP backbone and UnivFD head this codebase is written against.

UnivFD (Ojha, Li, Lee, CVPR 2023, "Towards Universal Fake Image Detectors that
Generalize Across Generative Models") is a single linear layer on top of frozen
OpenAI CLIP ViT-L/14 image features, trained on ProGAN real/fake images. The same
CLIP backbone also produces the embeddings the similarity search stores.

Both files are MIT licensed, but like TruFor they are fetched at setup time by
``scripts/fetch_univfd.py`` rather than committed: the backbone is 890 MB. Each is
checked against the SHA-256 below at install time and again before it is loaded.

Constants only, so the fetch script can import this module without pulling in
torch.
"""
from __future__ import annotations

UPSTREAM_REPO = "WisconsinAIVision/UniversalFakeDetect"
UPSTREAM_COMMIT = "763391e"
LICENCE_NOTE = (
    "UnivFD head: (c) 2023 WAIV, MIT (github.com/WisconsinAIVision/UniversalFakeDetect). "
    "CLIP ViT-L/14 weights: (c) 2021 OpenAI, MIT (github.com/openai/CLIP)."
)

#: OpenAI publishes each CLIP checkpoint under a URL containing its SHA-256.
CLIP_FILE = "ViT-L-14.pt"
CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
CLIP_SIZE_BYTES = 932_768_134
CLIP_URL = f"https://openaipublic.azureedge.net/clip/models/{CLIP_SHA256}/{CLIP_FILE}"

#: The upstream repository publishes no hash; this one was computed from the file
#: at the commit above (git blob 2a8efc8e238bf5a491e4fd5999655ed5eb3bbb72).
HEAD_FILE = "fc_weights.pth"
HEAD_SHA256 = "477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847"
HEAD_SIZE_BYTES = 4083
HEAD_URL = (
    "https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/"
    f"main/pretrained_weights/{HEAD_FILE}"
)

#: CLIP's image normalisation, which UnivFD's validation code also uses.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
INPUT_SIZE = 224
FEATURE_DIM = 768

MANIFEST_NAME = "manifest.json"
