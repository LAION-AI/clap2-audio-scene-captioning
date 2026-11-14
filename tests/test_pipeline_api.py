import numpy as np
import pytest
from fastapi.testclient import TestClient
from clap2_scenes.api import create_app
from clap2_scenes import SceneCaptioner, AnalysisConfig


class Encoder:
    model_size = "m"
    dimension = 2
    device = "cpu"
    spec = {"revision": "test"}


class Index:
    metadata = {
        "model_size": "m",
        "dimension": 2,
        "revision": "test",
        "index_size": "10k",
    }
    count = 10

    def retrieve(self, queries, top_n):
        return [
            [
                {"id": str(i), "text": f"Caption {i}", "score": float(q[0])}
                for i in range(top_n)
            ]
            for q in queries
        ]


def test_cached_features_and_validation():
    engine = SceneCaptioner(encoder=Encoder(), index=Index())
    f = {
        "model_size": "m",
        "model_revision": "test",
        "embeddings": np.tile([1.0, 0.0], (25, 1)),
        "bin_edges": np.arange(26) * 0.2,
        "analysis_method": "Sliding windows",
        "stride_seconds": 0.2,
        "window_seconds": 2,
        "window_count": 16,
    }
    result = engine.predict_features(f, AnalysisConfig(top_n=3))
    assert (
        len(result["timeline"]) == 25 and len(result["timeline"][0]["predictions"]) == 3
    )
    assert not result["audio_encoder_rerun"]
    with pytest.raises(ValueError, match="encoding audio again"):
        engine.predict_features(f, AnalysisConfig(stride_seconds=0.5))
    with pytest.raises(ValueError):
        engine.predict_features(f, AnalysisConfig(top_n=11))
    with pytest.raises(ValueError):
        engine.predict_features(f, AnalysisConfig(reencode_segments=True))
    with pytest.raises(ValueError):
        AnalysisConfig(top_n=True)


class FakePipeline:
    encoder = Encoder()
    index = Index()

    def __init__(self):
        self.paths = []

    def analyze_many(self, paths, config):
        self.paths = list(paths)
        assert all(p.is_file() for p in paths)
        if paths[0].read_bytes() == b"broken":
            raise ValueError("Invalid audio")
        return [
            {
                "source": p.name,
                "top_n": config.top_n,
                "method": config.segmentation.method,
            }
            for p in paths
        ]


def test_api_order_config_validation_and_cleanup():
    engine = FakePipeline()
    with TestClient(create_app(engine)) as c:
        assert c.get("/health").status_code == 200
        r = c.post(
            "/analyze",
            files=[("files", ("a.wav", b"a")), ("files", ("b.wav", b"b"))],
            data={"top_n": "3", "config": '{"segmentation":{"target_segments":3}}'},
        )
        assert r.status_code == 200, r.text
        assert [x["source"] for x in r.json()["results"]] == ["a.wav", "b.wav"]
        assert all(x["top_n"] == 3 for x in r.json()["results"])
        assert all(not p.exists() for p in engine.paths)
        for config in [
            "[]",
            "not json",
            '{"top_n":2}',
            '{"segmentation":{"method":"bad"}}',
        ]:
            assert (
                c.post(
                    "/analyze",
                    files={"files": ("a.wav", b"a")},
                    data={"config": config},
                ).status_code
                == 422
            )
        assert (
            c.post("/analyze", files={"files": ("a.wav", b"broken")}).status_code == 422
        )
        assert all(not p.exists() for p in engine.paths)


def test_upload_limit():
    with TestClient(create_app(FakePipeline(), max_upload_bytes=100)) as c:
        assert (
            c.post("/analyze", files={"files": ("a.wav", b"x" * 200)}).status_code
            == 413
        )
        # No Content-Length: also enforce the streaming request limit.
        r = c.post(
            "/analyze",
            content=iter([b"x" * 50, b"x" * 100]),
            headers={"content-type": "multipart/form-data; boundary=x"},
        )
        assert r.status_code == 413, r.text
