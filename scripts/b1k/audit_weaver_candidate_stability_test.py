from __future__ import annotations

import numpy as np
import pytest

from scripts.b1k import audit_weaver_candidate_stability as audit


def test_gripper_distance_report_exposes_tail_and_contract_failures() -> None:
    report = audit.gripper_distance_report(
        np.array([[1.0, -1.0], [0.96, -0.94], [0.75, -1.20]]),
        tolerance=0.10,
    )

    assert report["combined"]["count"] == 6
    assert report["combined"]["above_0p05_count"] == 3
    assert report["combined"]["above_0p10_count"] == 2
    assert report["combined"]["above_contract_tolerance_count"] == 2


def test_common_noise_report_separates_action_and_rollout_noise() -> None:
    scores = np.array(
        [
            [0.0, 2.0],
            [1.0, 3.0],
            [-1.0, 1.0],
        ]
    )
    report = audit.common_noise_score_report(scores)

    assert report["winner_indices"] == [1, 1, 1]
    assert report["winner_agreement_fraction"] == 1.0
    assert report["action_variance_under_common_noise"] == pytest.approx(1.0)
    assert report["rollout_noise_variance"] == pytest.approx(2 / 3)
    assert report["action_to_noise_rms_ratio"] == pytest.approx(np.sqrt(1.5))


def test_sampled_ordinals_are_unique_and_avoid_endpoints() -> None:
    assert audit._sampled_ordinals(20, 3) == (5, 10, 14)  # noqa: SLF001
    with pytest.raises(ValueError, match="unique ordinals"):
        audit._sampled_ordinals(2, 3)  # noqa: SLF001
