"""
Every image detector the application actually wires up has a signal family.

``_family`` fails safe -- an unlisted detector becomes "unclassified" and so
cannot corroborate independently -- but failing safe at runtime is not the same
as being correct. Two detectors added without family assignments would both land
in "unclassified", share one family, and silently stop corroborating each other
even though they read genuinely different signals. The only sign would be a log
warning nobody reads.

So the map is pinned against the detector list the container really builds. Add
an image detector without classifying it and this test names it.
"""
from app.services.detection.detection_container import create_detection_router
from app.services.detection.engine.image_detection_engine import (
    _NON_EVIDENTIAL,
    _SIGNAL_FAMILY,
)


def test_every_wired_image_detector_has_a_signal_family():
    router = create_detection_router()
    wired = [type(d).__name__ for d in router.image_engine.detectors]

    assert wired, "the container wired no image detectors at all"

    unclassified = [name for name in wired if name not in _SIGNAL_FAMILY]
    assert not unclassified, (
        "these image detectors reach ImageDetectionEngine with no signal family, "
        "so they will be pooled into 'unclassified' and cannot corroborate "
        f"independently: {unclassified}. Add each to _SIGNAL_FAMILY -- or to "
        f"'{_NON_EVIDENTIAL}' if its output is not manipulation evidence."
    )


def test_object_detectors_are_non_evidential():
    """
    Object presence must not be able to move an assessment in either direction.

    This is the specific prohibition on using object detection as proof an image
    is fake, pinned at the point that decides it.
    """
    for name, family in _SIGNAL_FAMILY.items():
        if "Object" in name:
            assert family == _NON_EVIDENTIAL, (
                f"{name} is classified as an evidential '{family}' signal; "
                "detecting what is in a photograph says nothing about whether it "
                "was manipulated"
            )


def test_families_that_share_a_physical_basis_are_grouped():
    """
    The groupings are the whole mechanism, so the load-bearing ones are asserted
    rather than left to a docstring.
    """
    # All read JPEG quantisation and block structure: one recompression makes
    # every one of them fire, which is one observation, not three.
    assert _SIGNAL_FAMILY["ForensicImageDetector"] == _SIGNAL_FAMILY[
        "ManipulationMaskGenerator"
    ]
    # Keypoint correspondence is independent of compression history.
    assert _SIGNAL_FAMILY["CopyMoveDetector"] != _SIGNAL_FAMILY[
        "ForensicImageDetector"
    ]
    # Both analyse facial regions; agreement between them is not independent.
    assert _SIGNAL_FAMILY["FaceRegionAnalyzer"] == _SIGNAL_FAMILY[
        "BaselineFaceDetector"
    ]
