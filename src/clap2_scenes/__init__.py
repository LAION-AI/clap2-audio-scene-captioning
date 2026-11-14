"""CLAP2 audio scene segmentation and nearest-neighbor caption retrieval."""

from .pipeline import SceneCaptioner, AnalysisConfig
from .segmentation import SegmentationConfig, segment_features

__all__ = ["SceneCaptioner", "AnalysisConfig", "SegmentationConfig", "segment_features"]
__version__ = "1.0.0"
