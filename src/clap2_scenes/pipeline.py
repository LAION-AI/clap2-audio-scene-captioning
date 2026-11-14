"""Single-audio and ordered-batch captioning with reusable audio features."""

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import math, time
import numpy as np
from .audio import AudioEncoder, load_audio, SAMPLE_RATE
from .index import CaptionIndex
from .segmentation import SegmentationConfig, segment_features
from .smoothing import smooth_scores


@dataclass(frozen=True)
class AnalysisConfig:
    window_seconds: float = 2.0
    stride_seconds: float = 0.2
    feature_method: str = "sliding"
    smoothing: str = "moving_average"
    smoothing_seconds: float = 1.0
    top_n: int = 5
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    include_embeddings: bool = False
    reencode_segments: bool = False

    def __post_init__(self):
        if isinstance(self.segmentation, dict):
            object.__setattr__(
                self, "segmentation", SegmentationConfig(**self.segmentation)
            )
        if not isinstance(self.segmentation, SegmentationConfig):
            raise ValueError(
                "segmentation must be SegmentationConfig or a configuration object"
            )
        if self.feature_method not in {"sliding", "contextual"}:
            raise ValueError("feature_method must be sliding or contextual")
        if self.smoothing not in {"none", "moving_average", "gaussian"}:
            raise ValueError("Unknown smoothing method")
        for key, lo, hi in [
            ("window_seconds", 0.5, 10),
            ("stride_seconds", 0.1, 1),
            ("smoothing_seconds", 0.01, 600),
        ]:
            value = getattr(self, key)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lo <= value <= hi
            ):
                raise ValueError(f"{key} must be finite and between {lo} and {hi}")
        if (
            self.feature_method == "sliding"
            and self.stride_seconds > self.window_seconds
        ):
            raise ValueError("stride_seconds cannot exceed window_seconds")
        if (
            isinstance(self.top_n, bool)
            or not isinstance(self.top_n, int)
            or self.top_n < 1
        ):
            raise ValueError("top_n must be a positive integer")
        if not isinstance(self.include_embeddings, bool) or not isinstance(
            self.reencode_segments, bool
        ):
            raise ValueError("Embedding options must be boolean")


class SceneCaptioner:
    """Load one CLAP2 model/index once, then annotate files or lists of files.

    `encode` and `predict_features` are separate so cached audio features can be
    reused for new segmentation, smoothing or Top-N settings without GPU encoding.
    """

    def __init__(
        self,
        model_size="m",
        index_size="100k",
        device=None,
        retrieval_device=None,
        cache_dir=None,
        model_cache_dir=None,
        memory_map=False,
        index_path=None,
        audio_batch_size=64,
        encoder=None,
        index=None,
    ):
        self.encoder = encoder or AudioEncoder(
            model_size, device, audio_batch_size, model_cache_dir
        )
        retrieval_device = retrieval_device or (
            "cpu" if memory_map else self.encoder.device
        )
        self.index = index or CaptionIndex(
            model_size, index_size, cache_dir, retrieval_device, memory_map, index_path
        )
        if (
            self.index.metadata["model_size"] != self.encoder.model_size
            or self.index.metadata["dimension"] != self.encoder.dimension
        ):
            raise ValueError("Audio encoder and text index must use the same model")
        if self.index.metadata["revision"] != self.encoder.spec["revision"]:
            raise ValueError("Audio and text checkpoint revisions differ")

    def encode(self, audio, config=None):
        config = config or AnalysisConfig()
        pcm, _ = load_audio(
            str(audio), max_seconds=30 if config.feature_method == "contextual" else 600
        )
        result = self.encoder.encode(
            pcm,
            config.window_seconds,
            config.stride_seconds,
            "Sliding windows"
            if config.feature_method == "sliding"
            else "Contextual patches",
        )
        result["source"] = Path(audio).name
        result["model_size"] = self.encoder.model_size
        result["model_revision"] = self.encoder.spec["revision"]
        return result

    def predict_features(self, features, config=None):
        config = config or AnalysisConfig()
        if config.reencode_segments:
            raise ValueError(
                "Fresh segment encoding requires original audio; use analyze()"
            )
        if (
            features["model_size"] != self.encoder.model_size
            or features["model_revision"] != self.encoder.spec["revision"]
        ):
            raise ValueError("Cached features belong to another model or checkpoint")
        if config.top_n > self.index.count:
            raise ValueError("top_n exceeds the candidate count")
        expected = (
            "Sliding windows"
            if config.feature_method == "sliding"
            else "Contextual patches"
        )
        if (
            features["analysis_method"] != expected
            or abs(features["stride_seconds"] - config.stride_seconds) > 1e-9
            or (
                expected == "Sliding windows"
                and abs(features["window_seconds"] - config.window_seconds) > 1e-9
            )
        ):
            raise ValueError(
                "Window, stride or feature-method changes require encoding audio again"
            )
        query = np.asarray(features["embeddings"], dtype="float32")
        edges = np.asarray(features["bin_edges"], dtype="float64")
        if len(query) != len(edges) - 1:
            raise ValueError("Cached feature rows do not match their time edges")
        method = {
            "none": "None",
            "moving_average": "Moving average",
            "gaussian": "Gaussian",
        }[config.smoothing]
        smoothed = smooth_scores(query, edges, method, config.smoothing_seconds)
        predictions = self.index.retrieve(smoothed, config.top_n)
        segmentation = segment_features(query, edges, config.segmentation)
        segment_queries = np.stack([s["embedding"] for s in segmentation["segments"]])
        segment_hits = self.index.retrieve(segment_queries, config.top_n)
        segments = []
        for i, (segment, hits) in enumerate(
            zip(segmentation.pop("segments"), segment_hits)
        ):
            vector = segment.pop("embedding")
            row = {
                "index": i,
                **segment,
                "duration": segment["end"] - segment["start"],
                "predictions": hits,
                "embedding_kind": "normalized_duration_weighted_cached_features",
            }
            if config.include_embeddings:
                row["embedding"] = vector.tolist()
            segments.append(row)
        result = {
            "schema_version": "1.0",
            "source": features.get("source"),
            "duration_seconds": float(edges[-1]),
            "model": {"size": self.encoder.model_size, **self.encoder.spec},
            "index": self.index.metadata,
            "config": asdict(config),
            "score_type": "cosine similarity or temporal mean; not calibrated probability",
            "timeline": [
                {"index": i, "start": float(a), "end": float(b), "predictions": hits}
                for i, (a, b, hits) in enumerate(
                    zip(edges[:-1], edges[1:], predictions)
                )
            ],
            "segments": segments,
            "segmentation": segmentation,
            "audio_encoder_rerun": False,
            "window_count": features["window_count"],
        }
        if config.include_embeddings:
            result["time_embeddings"] = query.tolist()
        return result

    def analyze(self, audio, config=None, *, top_n=None):
        config = config or AnalysisConfig()
        if top_n is not None:
            config = replace(config, top_n=top_n)
        start = time.perf_counter()
        features = self.encode(audio, config)
        encoded = time.perf_counter()
        result = self.predict_features(
            features, replace(config, reencode_segments=False)
        )
        result["config"] = asdict(config)
        if config.reencode_segments:
            pcm, _ = load_audio(str(audio))
            fresh = []
            for segment in result["segments"]:
                if segment["duration"] > 30 + 1e-8:
                    raise ValueError(
                        "Fresh cut encoding supports segments up to 30 seconds; use more segments or cached feature captions"
                    )
                a = round(segment["start"] * SAMPLE_RATE)
                b = round(segment["end"] * SAMPLE_RATE)
                fresh.append(self.encoder._preprocess_windows(pcm[a:b], [0], b - a)[0])
            hits = self.index.retrieve(np.stack(fresh), config.top_n)
            for segment, vector, predictions in zip(result["segments"], fresh, hits):
                segment["predictions"] = predictions
                segment["embedding_kind"] = "fresh_isolated_audio_cut"
                if config.include_embeddings:
                    segment["embedding"] = vector.tolist()
        result["audio_encoder_rerun"] = True
        result["timing"] = {
            "audio_encoding_seconds": encoded - start,
            "retrieval_and_segmentation_seconds": time.perf_counter() - encoded,
            "total_seconds": time.perf_counter() - start,
        }
        return result

    def analyze_many(self, audios, config=None, *, top_n=None):
        return [self.analyze(audio, config, top_n=top_n) for audio in audios]

    def __call__(self, audio, config=None, *, top_n=None):
        return (
            self.analyze_many(audio, config, top_n=top_n)
            if isinstance(audio, (list, tuple))
            else self.analyze(audio, config, top_n=top_n)
        )
