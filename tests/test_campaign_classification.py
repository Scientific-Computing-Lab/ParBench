"""TDD stubs for Phase 2's campaign_for() classification function.

These tests define the interface contract for campaign_for(), which will
classify evaluation results into Campaign 1 (C1: deterministic, temp=0)
or Campaign 2 (C2: stochastic, temp>0, multiple samples).

The function does not exist yet -- tests are skipped until Phase 2
implements scripts.evaluation.campaign_utils.campaign_for().

See decisions D-16 through D-20 in 01-CONTEXT.md.
"""

from __future__ import annotations

import pytest

# Safe import: prevents ImportError during collection (Pitfall 6 from RESEARCH.md).
try:
    from scripts.evaluation.campaign_utils import campaign_for
except ImportError:
    campaign_for = None  # type: ignore[assignment]


@pytest.mark.skip(reason="campaign_for() not yet implemented -- Phase 2")
def test_c1_classification() -> None:
    """C1: deterministic eval (temp=0, no sample_id). Per D-17, D-19."""
    result = campaign_for({"temperature": 0.0, "sample_id": None})
    assert result == "c1"


@pytest.mark.skip(reason="campaign_for() not yet implemented -- Phase 2")
def test_c2_classification() -> None:
    """C2: stochastic eval (temp=0.7, sample_id present). Per D-18, D-19."""
    result = campaign_for({"temperature": 0.7, "sample_id": "s1"})
    assert result == "c2"
