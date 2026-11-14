"""Duration-aware temporal weighting, equivalent to smoothing every candidate score."""

from typing import Any
import numpy as np
from .audio import _finite_float


def smooth_scores(scores: Any, bin_edges: Any, method: str, width: float) -> np.ndarray:
    """Smooth per-bin class scores using duration-aware sparse weights."""
    matrix = np.asarray(scores)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a nonempty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores contain non-finite values")
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or edges.size != matrix.shape[0] + 1:
        raise ValueError("bin_edges must have length N + 1")
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("bin_edges must be finite and strictly increasing")
    normalized_method = str(method).strip().lower()
    if normalized_method in {"none", ""}:
        return np.array(matrix, copy=True)
    if normalized_method not in {"moving average", "gaussian"}:
        raise ValueError("method must be None, Moving average, or Gaussian")
    width_seconds = _finite_float(width, "width")
    if width_seconds <= 0:
        raise ValueError("width must be positive for score smoothing")

    from scipy.sparse import csr_matrix

    starts = edges[:-1]
    ends = edges[1:]
    centers = (starts + ends) * 0.5
    durations = ends - starts
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    if normalized_method == "gaussian":
        sigma = width_seconds / 2.35482
        radius = 3.0 * sigma
    else:
        radius = width_seconds * 0.5

    for i, center in enumerate(centers):
        left = center - radius
        right = center + radius
        first = max(0, int(np.searchsorted(ends, left, side="right")))
        last = min(len(centers), int(np.searchsorted(starts, right, side="left")))
        js = np.arange(first, max(first, last), dtype=np.int64)
        if normalized_method == "moving average":
            weights = np.maximum(
                0.0,
                np.minimum(ends[js], right) - np.maximum(starts[js], left),
            )
        else:
            weights = durations[js] * np.exp(
                -0.5 * ((centers[js] - center) / sigma) ** 2
            )
        keep = weights > 0
        js = js[keep]
        weights = weights[keep]
        total = float(weights.sum())
        if total <= 0:
            js = np.array([i], dtype=np.int64)
            weights = np.array([1.0], dtype=np.float64)
            total = 1.0
        rows.extend([i] * len(js))
        columns.extend(js.tolist())
        values.extend((weights / total).tolist())

    weight_matrix = csr_matrix(
        (values, (rows, columns)), shape=(len(centers), len(centers))
    )
    result = weight_matrix @ matrix.astype(np.float64, copy=False)
    return np.asarray(
        result,
        dtype=matrix.dtype if np.issubdtype(matrix.dtype, np.floating) else np.float32,
    )
