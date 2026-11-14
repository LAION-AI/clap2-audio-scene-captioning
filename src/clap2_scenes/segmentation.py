"""Contiguous segmentation of cached audio features; no learned event detector."""

from dataclasses import dataclass, asdict
import heapq, math
import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    method: str = "hierarchical"
    target_segments: int = 5
    min_segment_seconds: float = 1.0
    grid_seconds: float = 0.0
    sensitivity: float = 1.0
    spread_estimator: str = "robust"
    statistics_window_seconds: float = 0.0
    feature_smoothing_seconds: float = 0.0
    peak_selection: str = "local_maxima"
    minimum_peak_contrast: float = 0.0
    fixed_distance: float = 0.05
    percentile: float = 85.0
    novelty_neighborhood_seconds: float = 0.8

    def __post_init__(self):
        if self.method not in {
            "hierarchical",
            "adaptive",
            "fixed",
            "percentile",
            "novelty",
        }:
            raise ValueError("Unknown segmentation method")
        if (
            isinstance(self.target_segments, bool)
            or not isinstance(self.target_segments, int)
            or not 1 <= self.target_segments <= 6000
        ):
            raise ValueError("target_segments must be an integer between 1 and 6000")
        limits = {
            "min_segment_seconds": (0.1, 600),
            "grid_seconds": (0, 600),
            "sensitivity": (-10, 20),
            "statistics_window_seconds": (0, 600),
            "feature_smoothing_seconds": (0, 600),
            "minimum_peak_contrast": (0, 2),
            "fixed_distance": (0, 2),
            "percentile": (0, 100),
            "novelty_neighborhood_seconds": (0.01, 600),
        }
        for key, (lo, hi) in limits.items():
            v = getattr(self, key)
            if (
                not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not lo <= v <= hi
            ):
                raise ValueError(f"{key} must be finite and between {lo} and {hi}")
        if self.spread_estimator not in {"robust", "mad", "iqr"}:
            raise ValueError("spread_estimator must be robust, mad, or iqr")
        if self.peak_selection not in {"local_maxima", "crossings", "all"}:
            raise ValueError("peak_selection must be local_maxima, crossings, or all")


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x), where=n > 1e-12).astype("float32")


def aggregate(rows, edges, left, right):
    first = max(0, np.searchsorted(edges[1:], left, side="right"))
    last = min(len(rows), np.searchsorted(edges[:-1], right, side="left"))
    weights = np.maximum(
        0,
        np.minimum(edges[first + 1 : last + 1], right)
        - np.maximum(edges[first:last], left),
    )
    if not weights.size or weights.sum() <= 0:
        raise ValueError("Interval must overlap at least one feature bin")
    return (
        weights @ np.asarray(rows[first:last], dtype=np.float64) / weights.sum()
    ).astype("float32")


def robust(values, sensitivity=1.0, estimator="robust"):
    values = np.asarray(values, dtype="float64")
    if not len(values):
        return {
            "median": 0.0,
            "spread": 1e-6,
            "threshold": max(1e-6, sensitivity * 1e-6),
        }
    med = float(np.median(values))
    mad = 1.4826 * float(np.median(abs(values - med)))
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
    spread = max(
        1e-6,
        mad if estimator == "mad" else iqr if estimator == "iqr" else max(mad, iqr),
    )
    return {
        "median": med,
        "spread": spread,
        "threshold": max(1e-6, med + sensitivity * spread),
    }


def adjacent_hierarchy(rows, edges, target_segments=5, min_segment_seconds=1.0):
    """Adjacent-only, duration-weighted Ward-like merging, with stable leftmost ties."""
    n = len(rows)
    means = [np.array(v, dtype="float64") for v in rows]
    weights = np.diff(edges).copy()
    ends = np.arange(1, n + 1)
    previous = np.arange(n) - 1
    following = np.arange(n) + 1
    following[-1] = -1
    active = np.ones(n, dtype=bool)
    versions = np.zeros(n, dtype=int)
    heap = []

    def cost(i, j):
        delta = means[i] - means[j]
        return float(
            weights[i] * weights[j] / (weights[i] + weights[j]) * np.dot(delta, delta)
        )

    def push(i):
        if i >= 0 and active[i] and following[i] >= 0:
            j = int(following[i])
            heapq.heappush(heap, (cost(i, j), i, j, int(versions[i]), int(versions[j])))

    for i in range(n - 1):
        push(i)

    def merge(i):
        j = int(following[i])
        w = weights[i] + weights[j]
        means[i] = (means[i] * weights[i] + means[j] * weights[j]) / w
        weights[i] = w
        ends[i] = ends[j]
        active[j] = False
        versions[i] += 1
        following[i] = following[j]
        if following[i] >= 0:
            previous[following[i]] = i
        push(int(previous[i]))
        push(i)

    count = n
    while count > max(1, target_segments):
        _, i, j, vi, vj = heapq.heappop(heap)
        if (
            not active[i]
            or not active[j]
            or following[i] != j
            or versions[i] != vi
            or versions[j] != vj
        ):
            continue
        merge(i)
        count -= 1
    while count > 1:
        short = next(
            (
                i
                for i in range(n)
                if active[i] and weights[i] < min_segment_seconds - 1e-8
            ),
            None,
        )
        if short is None:
            break
        left = int(previous[short])
        right = int(following[short])
        i = (
            short
            if left < 0
            else left
            if right < 0 or cost(left, short) <= cost(short, right)
            else short
        )
        merge(i)
        count -= 1
    return [0] + [int(ends[i]) for i in range(n) if active[i]]


def select_boundaries(scores, edges, threshold, min_duration, rule, contrast):
    candidates = []
    above = 0
    for i in range(1, len(scores)):
        if not scores[i] > max(1e-6, threshold[i]):
            continue
        above += 1
        previous = scores[i - 1]
        following = scores[i + 1] if i + 1 < len(scores) else 0
        if rule == "local_maxima" and not (
            scores[i] >= previous and scores[i] > following
        ):
            continue
        if rule == "crossings" and i > 1 and scores[i - 1] > threshold[i - 1]:
            continue
        if contrast > 0 and scores[i] - (previous + following) / 2 < contrast - 1e-12:
            continue
        candidates.append(i)
    candidates.sort(key=lambda i: (-scores[i], i))
    chosen = [0, len(edges) - 1]
    for i in candidates:
        if all(abs(edges[i] - edges[j]) >= min_duration - 1e-8 for j in chosen):
            chosen.append(i)
    return sorted(chosen), {
        "bins_above_threshold": above,
        "peak_candidates": len(candidates),
        "accepted_boundaries": len(chosen) - 2,
    }


def segment_features(features, bin_edges, config=None):
    config = config or SegmentationConfig()
    raw = np.asarray(features, dtype="float32")
    original_edges = np.asarray(bin_edges, dtype="float64")
    if raw.ndim != 2 or not len(raw) or not raw.shape[1] or not np.isfinite(raw).all():
        raise ValueError("features must be a nonempty finite matrix")
    if (
        original_edges.shape != (len(raw) + 1,)
        or not np.isfinite(original_edges).all()
        or original_edges[0] != 0
        or np.any(np.diff(original_edges) <= 0)
    ):
        raise ValueError(
            "bin_edges must increase from zero with one more entry than features"
        )
    duration = float(original_edges[-1])
    native = float(np.median(np.diff(original_edges)))
    step = max(native, config.grid_seconds)
    edges = [0.0]
    while edges[-1] + step < duration - 1e-7:
        edges.append(edges[-1] + step)
    edges = np.array(edges + [duration])
    grid = np.stack(
        [aggregate(raw, original_edges, a, b) for a, b in zip(edges[:-1], edges[1:])]
    )
    rows = normalize(grid)
    if config.feature_smoothing_seconds > 0:
        radius = config.feature_smoothing_seconds / 2
        rows = normalize(
            np.stack(
                [
                    aggregate(
                        grid, edges, max(0, c - radius), min(duration, c + radius)
                    )
                    for c in (edges[:-1] + edges[1:]) / 2
                ]
            )
        )
    n = len(rows)
    distance = np.zeros(n)
    novelty = np.zeros(n)
    for i in range(1, n):
        distance[i] = np.clip(
            1 - np.dot(rows[i - 1].astype("float64"), rows[i].astype("float64")), 0, 2
        )
        t = edges[i]
        left = aggregate(
            rows, edges, max(0, t - config.novelty_neighborhood_seconds), t
        )
        right = aggregate(
            rows, edges, t, min(duration, t + config.novelty_neighborhood_seconds)
        )
        novelty[i] = np.sum((left.astype("float64") - right) ** 2) / 4
    stats = robust(distance[1:], config.sensitivity, config.spread_estimator)
    novel_stats = robust(novelty[1:], config.sensitivity, config.spread_estimator)
    threshold = None
    diagnostics = None
    if config.method == "hierarchical":
        boundaries = adjacent_hierarchy(
            rows, edges, config.target_segments, config.min_segment_seconds
        )
    else:
        scores = novelty if config.method == "novelty" else distance
        if config.method == "fixed":
            threshold = np.full(n, config.fixed_distance)
        elif config.method == "percentile":
            threshold = np.full(
                n,
                max(
                    1e-6,
                    float(np.quantile(distance[1:], config.percentile / 100))
                    if n > 1
                    else 0,
                ),
            )
        else:
            threshold = np.full(
                n, (novel_stats if config.method == "novelty" else stats)["threshold"]
            )
            if config.statistics_window_seconds:
                half = max(
                    2, math.floor(config.statistics_window_seconds / step / 2 + 0.5)
                )
                threshold = np.array(
                    [
                        robust(
                            scores[max(1, i - half) : min(n, i + half + 1)],
                            config.sensitivity,
                            config.spread_estimator,
                        )["threshold"]
                        for i in range(n)
                    ]
                )
        boundaries, diagnostics = select_boundaries(
            scores,
            edges,
            threshold,
            config.min_segment_seconds,
            config.peak_selection,
            config.minimum_peak_contrast,
        )
    segments = [
        {
            "start": float(edges[a]),
            "end": float(edges[b]),
            "embedding": normalize(aggregate(raw, original_edges, edges[a], edges[b])),
        }
        for a, b in zip(boundaries[:-1], boundaries[1:])
    ]
    return {
        "segments": segments,
        "method": config.method,
        "parameters": asdict(config),
        "grid_seconds": step,
        "grid_edges": edges.tolist(),
        "boundary_indices": boundaries,
        "adjacent_distances": distance.tolist(),
        "novelty_scores": novelty.tolist(),
        "thresholds": None if threshold is None else threshold.tolist(),
        "neighbor_statistics": stats,
        "novelty_statistics": novel_stats,
        "diagnostics": diagnostics,
        "clip_shorter_than_minimum": duration < config.min_segment_seconds,
    }
