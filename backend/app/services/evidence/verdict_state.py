"""
Whether a result's verdicts may be read as conclusions or only as advice.

Every verdict is ADVISORY until the detectors and the rules that combine them
pass independent evaluation on data they were never fitted or tuned on (the
frozen holdout, ``backend/evaluation/holdout.lock.json``). The state is recorded
in each analysis's verdict policy at analysis time, so a stored result keeps
the state it was produced under; results recorded before the state existed
are ADVISORY, because no verdict had passed evaluation then.
"""
from typing import Any, Optional

ADVISORY = "ADVISORY"
VALIDATED = "VALIDATED"

STATEMENTS = {
    ADVISORY: (
        "Experimental, advisory result. AIMD's verdicts have not yet passed "
        "independent evaluation, so this result supports an examiner's review "
        "and must not be relied on as a forensic conclusion on its own."
    ),
    VALIDATED: (
        "Produced under a verdict policy that passed independent evaluation. "
        "The stated limitations and likelihood-ratio ranges still apply."
    ),
}


def verdict_state(config_snapshot: Optional[dict[str, Any]]) -> dict[str, str]:
    policy = (config_snapshot or {}).get("verdict_policy") or {}
    state = policy.get("release_state")
    if state not in STATEMENTS:
        state = ADVISORY
    return {"state": state, "statement": STATEMENTS[state]}
