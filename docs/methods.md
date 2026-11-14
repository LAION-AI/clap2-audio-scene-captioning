# From audio to contiguous scenes

The pipeline has two related outputs: a dense timeline of caption matches and a coarser sequence of contiguous segments. Both use the same audio features. Segment boundaries depend on the audio features, not the retrieved caption names.

## 1. Decode and encode

FFmpeg decodes one complete audio track to mono float32 PCM at 32 kHz. It reads through the end rather than relying on potentially inaccurate container duration metadata. Sliding mode accepts at most 600 seconds; contextual mode accepts at most 30 seconds. Overlong audio is rejected, never silently truncated.

### Sliding windows (default)

A 2-second window moves by 0.2 seconds. If the final regular window does not reach the end, an additional end-aligned window covers the tail. Audio shorter than the requested window is encoded once. The underlying CLAP2 preprocessing handles patch padding.

Let `u_j` be the normalized CLAP2 embedding of window `j`, and let `w_tj` be its duration of overlap with output bin `t`. The time query is:

```text
q_t = sum_j(w_tj × u_j) / sum_j(w_tj)
score(t, caption) = dot(q_t, normalized_text_embedding)
```

We do not normalize `q_t` again. Consequently, the score equals the overlap-weighted mean of the original window/text cosines. Normalizing the mean would change the meaning and scale of the score.

The last output bin ends at the exact decoded duration and may be shorter than the configured stride. Every sample belongs to the covered time range.

### Contextual patches (experimental)

Set `feature_method="contextual"`. The full recording passes through the audio transformer once. For each time bin, only final hidden audio patches overlapping that interval enter the pretrained attention pooling and projection head. Register/prefix tokens and all non-overlapping patches are excluded. Partial patch overlap contributes a logarithmic attention prior proportional to the fraction of that patch inside the bin. Other patches are not zero-valued inputs to pooling; they are excluded from its softmax support.

The selected hidden states already contain context from the full transformer pass. Their embeddings therefore include information about surrounding audio. Native patch time resolution is approximately 160 ms; a 200 ms output grid is not an independently trained 200 ms detector. This feature path relies on the pinned OpenCLIP architecture and is limited to 30 seconds to bound attention memory. `window_seconds` has no effect in contextual mode.

## 2. Smooth predictions without a full score matrix

`AnalysisConfig.smoothing` is one of:

| Setting | Meaning |
|---|---|
| `none` | Use each time query as-is |
| `moving_average` | Centered window of width `smoothing_seconds`; weight each bin by overlap duration |
| `gaussian` | Gaussian kernel with FWHM `smoothing_seconds`, approximately truncated at ±3 sigma, weighted by bin duration |

For Gaussian smoothing, bins intersecting the ±3-sigma support are included and evaluated at their centers. All kernels are renormalized at clip edges. The default is a 1-second moving average.

Smoothing feature vectors before dot products is algebraically equivalent to smoothing all candidate scores before Top-N selection. It is **not** smoothing only the previously returned Top 5, which could discard a candidate that should become a top match after smoothing. Averaged queries remain unnormalized.

## 3. Segment the feature sequence

`SegmentationConfig` controls boundary detection. The original features are duration-averaged onto a grid (native output stride by default), then normalized to unit vectors. Optional *feature smoothing* affects boundary detection only. It is independent of the prediction smoothing above.

### Adjacent hierarchical clustering — default

Start with one group per feature bin. Repeatedly merge the pair of **adjacent** groups with the smallest cost:

```text
cost(A, B) = duration(A) × duration(B) / (duration(A) + duration(B))
             × ||mean(A) − mean(B)||²
```

The group means are duration-weighted means of the initially normalized features; merged means are not normalized again. The cost is a duration-weighted Ward-like increase in squared error. Adjacency ensures every group is a contiguous interval, unlike unconstrained k-means.

Stop at `target_segments` (default **5**). Then scan for groups shorter than `min_segment_seconds` (default **1.0**) and merge each with its lower-cost neighbor. This second phase can reduce the final count. Exact cost ties choose the leftmost pair. The Python implementation uses a priority queue to avoid rescanning all possible adjacent merges at every step.

A whole clip shorter than the minimum remains one segment and sets `clip_shorter_than_minimum=true`. The minimum is otherwise respected. The requested count is a control over granularity, not an estimate of the true number of sound events.

### Adaptive neighbor changes

Calculate `d_t = 1 − cosine(x_(t−1), x_t)` between adjacent normalized time features. Large **distance**, equivalently low cosine similarity, suggests a change.

The default adaptive threshold is:

```text
median(d) + sensitivity × spread(d)
spread = max(1e-6, 1.4826 × MAD, IQR / 1.349)
```

`spread_estimator="mad"` or `"iqr"` selects just that robust estimate. Thresholds are floored at `1e-6`. By default the statistics use the full recording. Set `statistics_window_seconds` to use local statistics around each potential boundary; at least two neighboring bins per side are considered where available.

**Why can it return only one segment?** Overlapping windows make adjacent embeddings similar, so distances can be small. The adaptive threshold, peak filter, or minimum duration may then reject every boundary. Output diagnostics distinguish these cases.

For more changes, try `sensitivity=0.5`, `spread_estimator="mad"`, `statistics_window_seconds=3`, or a smaller minimum duration. Detection smoothing can further suppress changes; start with `feature_smoothing_seconds=0`. Lower sensitivity lowers the threshold; it does not guarantee an event count.

### Fixed neighbor distance

Use the same adjacent distance and a fixed `fixed_distance` cutoff (default **0.05**). This corresponds to cosine similarity below 0.95, subject to peak selection and minimum duration. Useful for a direct, interpretable control, but the same cutoff need not work for every recording, window length or model.

### Percentile neighbor peaks

Set the cutoff to the selected percentile of adjacent distances (default **85**). Lower percentiles admit more candidate changes. This adapts to scale without using a robust spread estimate, but it can promote weak fluctuations in a nearly stationary recording. The threshold floor and minimum duration still apply.

### Neighborhood novelty

Compare duration-weighted means of normalized features on each side of a potential boundary:

```text
novelty(t) = ||left_mean(t) − right_mean(t)||² / 4
```

Each side spans `novelty_neighborhood_seconds` (default **0.8**). The means are not renormalized. Apply the same robust adaptive thresholding as above, now to novelty values. This detects broader acoustic transitions rather than only a change between two immediately adjacent bins.

### Peak filtering and spacing

All four threshold-based methods share:

1. A score must strictly exceed the threshold.
2. `peak_selection="local_maxima"` (default) keeps scores at least as large as the previous score and strictly larger than the next. `"crossings"` keeps the beginning of an above-threshold region. `"all"` keeps every above-threshold candidate.
3. `minimum_peak_contrast` optionally requires a score to exceed the mean of its two neighboring scores by the specified amount.
4. Candidates are sorted strongest first. A candidate is accepted only if it is at least the minimum segment duration away from all accepted boundaries and both clip ends.

Diagnostics report the number of bins above threshold, peak candidates and accepted boundaries. Hierarchical clustering does not use these thresholds or peak rules.

## Every segmentation setting

| Parameter | Default | Used by |
|---|---:|---|
| `method` | `hierarchical` | All; alternatives `adaptive`, `fixed`, `percentile`, `novelty` |
| `target_segments` | 5 | Hierarchical |
| `min_segment_seconds` | 1.0 | All |
| `grid_seconds` | 0.0 | All; 0 means native stride, values below native stride do not upsample |
| `feature_smoothing_seconds` | 0.0 | All; centered duration-weighted smoothing before detection |
| `sensitivity` | 1.0 | Adaptive / novelty |
| `spread_estimator` | `robust` | Adaptive / novelty; also `mad`, `iqr` |
| `statistics_window_seconds` | 0.0 | Adaptive / novelty; 0 means whole clip |
| `fixed_distance` | 0.05 | Fixed |
| `percentile` | 85.0 | Percentile |
| `novelty_neighborhood_seconds` | 0.8 | Novelty; width on each side |
| `peak_selection` | `local_maxima` | All threshold methods |
| `minimum_peak_contrast` | 0.0 | All threshold methods |

## 4. Caption each segment

The default segment query is the normalized duration-weighted mean of the **original cached time features** inside the span. This is efficient and does not rerun the encoder. Sliding-window features near a boundary can include audio from outside the final segment.

For an embedding based only on the cut waveform, set `AnalysisConfig(reencode_segments=True)`. Every segment is decoded from the original recording's PCM and passed through CLAP2 as its own clip, then matched against the same full candidate library. Segments longer than 30 seconds are rejected in this mode. `predict_features()` cannot perform this operation because it has no original waveform.

In either mode, segment matches are freshly retrieved from the whole index. They are not a vote among the per-bin Top 5.

## What this is not

This is an offline, heuristic segmentation and retrieval pipeline. It does not train an event-boundary model, isolate sound sources, produce calibrated event probabilities, or guarantee that candidate descriptions are correct. Overlapping events can share one segment and multiple plausible captions. The 10k/50k/100k *caption-library clustering* was a separate offline spherical E-means operation over 100M text embeddings; it is not the adjacent hierarchical clustering used on a new audio recording.
