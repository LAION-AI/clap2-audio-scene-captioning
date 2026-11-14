import json
from pathlib import Path
import numpy as np
import pytest
from clap2_scenes.segmentation import (
    SegmentationConfig,
    segment_features,
    adjacent_hierarchy,
)

DATA = json.loads(
    (Path(__file__).parent / "fixtures/space-segmentation.json").read_text()
)


@pytest.mark.parametrize("case", DATA["cases"])
def test_space_algorithm_parity(case):
    o = case["options"]
    c = SegmentationConfig(
        method="fixed" if o["method"] == "manual" else o["method"],
        target_segments=o["count"],
        grid_seconds=o["grid"],
        min_segment_seconds=o["minDuration"],
        sensitivity=o["sigma"],
        spread_estimator=o["scale"],
        statistics_window_seconds=o["adaptiveWindow"],
        feature_smoothing_seconds=o["smooth"],
        peak_selection="local_maxima" if o["selection"] == "peaks" else o["selection"],
        minimum_peak_contrast=o["contrast"],
        fixed_distance=o["cutoff"],
        percentile=o["percentile"],
        novelty_neighborhood_seconds=o["neighborhood"],
    )
    r = segment_features(DATA["features"], DATA["edges"], c)
    assert r["boundary_indices"] == case["boundaries"]
    np.testing.assert_allclose(r["adjacent_distances"], case["distances"], atol=2e-7)
    np.testing.assert_allclose(r["novelty_scores"], case["novelty"], atol=2e-7)
    np.testing.assert_allclose(
        [s["embedding"] for s in r["segments"]],
        [s["vector"] for s in case["segments"]],
        atol=2e-7,
    )


@pytest.mark.parametrize(
    "method", ["hierarchical", "adaptive", "fixed", "percentile", "novelty"]
)
def test_coverage_and_minimum(method):
    rng = np.random.default_rng(7)
    edges = np.r_[np.arange(97) * 0.2, 19.31]
    r = segment_features(
        rng.normal(size=(97, 8)),
        edges,
        SegmentationConfig(method=method, target_segments=30),
    )
    segments = r["segments"]
    assert segments[0]["start"] == 0 and segments[-1]["end"] == edges[-1]
    assert all(s["end"] - s["start"] >= 1 - 1e-8 for s in segments)
    assert all(a["end"] == b["start"] for a, b in zip(segments, segments[1:]))


def test_short_constant_and_invalid():
    r = segment_features([[1, 0]], [0, 0.02])
    assert len(r["segments"]) == 1 and r["clip_shorter_than_minimum"]
    assert adjacent_hierarchy(np.ones((5, 3)), np.arange(6), 2, 1) == [0, 4, 5]
    assert (
        len(
            segment_features(
                np.ones((50, 3)),
                np.arange(51) * 0.2,
                SegmentationConfig(method="adaptive"),
            )["segments"]
        )
        == 1
    )
    for rows, edges in [([[float("nan")]], [0, 1]), ([[1], [2]], [0, 1, 1]), ([], [])]:
        with pytest.raises(ValueError):
            segment_features(rows, edges)
    with pytest.raises(ValueError):
        SegmentationConfig(min_segment_seconds=float("nan"))
