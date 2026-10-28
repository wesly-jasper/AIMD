"""Calibration coverage (the image sizes it was fitted on) and its bootstrap refits."""
import numpy as np

from app.services.detection import calibration


def _scores(n=200, seed=1):
    rng = np.random.default_rng(seed)
    targets = rng.random(n) < 0.5
    scores = np.clip(np.where(targets, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0.01, 0.99)
    return scores.tolist(), targets.tolist()


def test_coverage_is_a_low_percentile_of_the_short_sides():
    sides = list(range(100, 1100, 10))  # 100 images, 100..1090 px
    cov = calibration._coverage(sides + [None])
    assert cov["min_short_side"] == int(np.percentile(sides, calibration.COVERAGE_PERCENTILE))
    assert cov["images_with_size"] == 100
    assert calibration._coverage([None, None]) == {}


def test_bootstrap_draws_whole_groups():
    scores, targets = _scores()
    seen = []

    def fit_one(s, t):
        seen.append(len(s))
        return calibration.fit_platt(s, t)

    # Every image appears as four copies in one group: a resample holds whole groups.
    groups = [i // 4 for i in range(len(scores))]
    calibration._bootstrap(fit_one, scores, targets, groups, n=20)
    assert seen and all(n % 4 == 0 for n in seen)


def test_fit_records_coverage_and_refits_and_they_survive_a_save(tmp_path):
    scores, targets = _scores()
    cal = calibration.fit("BFreeDetector", "v", scores, targets, target_fpr=0.05, fitted_on={},
                          short_sides=[512] * len(scores), bootstrap=50)
    assert cal.coverage["min_short_side"] == 512
    assert 40 <= len(cal.bootstrap) <= 50
    slopes = [a for a, _ in cal.bootstrap]
    assert min(slopes) < cal.a < max(slopes)
    calibration.save({"BFreeDetector": cal}, tmp_path / "c.json")
    back = calibration.load(tmp_path / "c.json")["BFreeDetector"]
    assert back.coverage == cal.coverage and back.bootstrap == cal.bootstrap


def test_flag_rate_fit_has_refits_too():
    scores, targets = _scores()
    cal = calibration.fit_flag_rate("CopyMoveDetector", "v", scores, targets, flag_threshold=0.5,
                                    fitted_on={}, bootstrap=30)
    assert len(cal.bootstrap) == 30 and cal.coverage == {}
