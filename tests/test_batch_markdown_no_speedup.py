"""Rule 29 enforcement: the batch markdown summary must not render speedup_ratio.

Wall-clock speedup ratios from sub-millisecond baselines are unreliable
(nn=16.0x, bfs=0.002x); valid speedups come from profilers only. The result
JSONs keep the field for historical shape stability; the summary renderer is
the one consumer and must drop it.
"""

from scripts.evaluation.run_eval_batch import _generate_markdown

RECORD = {
    "overall_status": "PASS",
    "kernel": "nn",
    "augment_level": 0,
    "speedup_ratio": 16.0,
    "timing_method": "wall_time",
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "model": "test-model",
}


def test_markdown_has_no_speedup_column():
    md = _generate_markdown([RECORD], ["test-model"], "test batch")
    assert "Speedup" not in md
    assert "16.000" not in md
    assert "×" not in md  # the multiplication sign the old cell rendered
    # The Timing Method column stays (it is not a derived ratio).
    assert "Timing Method" in md
