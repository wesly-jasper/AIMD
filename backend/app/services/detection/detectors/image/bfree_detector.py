from app.services.detection.detectors.image.univfd_detector import UnivFDDetector
from app.services.detection.models.bfree_runner import LIMITATIONS, SEMANTICS


class BFreeDetector(UnivFDDetector):
    """
    Whole-image AI-generation detector (B-Free, CVPR 2025).

    The same contract as UnivFD -- one score against a threshold, reported on
    the ``ai_generation`` branch and never folded into the manipulation verdict
    -- from a different model. With both configured, the branch reports each,
    and a disagreement as CONFLICTING.

    The threshold applies to ``sigmoid(logit)``; 0.5 is the authors' decision
    rule ``logit > 0``.
    """

    detector_id = "BFreeDetector"
    version = "1.0.0"
    model_label = "B-Free"
    semantics = SEMANTICS
    limitations = tuple(LIMITATIONS)
