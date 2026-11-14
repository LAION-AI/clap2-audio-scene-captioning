"""Opt-in integration check using the release assets and all five real models."""

import gc, json, os, time
from pathlib import Path
import numpy as np
from clap2_scenes import SceneCaptioner, AnalysisConfig
from clap2_scenes.index import CaptionIndex
from clap2_scenes.segmentation import SegmentationConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch

    reports = []
    for model in ("xxs", "xs", "s", "m", "l"):
        start = time.perf_counter()
        engine = SceneCaptioner(
            model,
            index_path=ROOT / f"artifacts/clap2-{model}-luna-100k.npy",
            device=os.getenv("CLAP2_TEST_DEVICE", "cuda"),
            model_cache_dir=os.getenv("CLAP2_MODEL_CACHE"),
        )
        cfg = AnalysisConfig()
        features = engine.encode(ROOT / "examples/synthetic-scene.wav", cfg)
        output = engine.predict_features(features, cfg)
        assert (
            len(output["timeline"]) == 60
            and abs(output["segments"][-1]["end"] - 12) < 1e-6
        )
        assert all(s["duration"] >= 1 - 1e-7 for s in output["segments"])
        assert all(len(b["predictions"]) == 5 for b in output["timeline"])
        alt = engine.predict_features(
            features,
            AnalysisConfig(
                top_n=3,
                smoothing="gaussian",
                segmentation=SegmentationConfig(target_segments=3),
            ),
        )
        assert (
            len(alt["timeline"][0]["predictions"]) == 3
            and not alt["audio_encoder_rerun"]
        )
        ctx = engine.analyze(
            ROOT / "examples/synthetic-scene.wav",
            AnalysisConfig(feature_method="contextual"),
        )
        assert len(ctx["timeline"]) == 60
        fresh = engine.analyze(
            ROOT / "examples/synthetic-scene.wav",
            AnalysisConfig(
                reencode_segments=True,
                segmentation=SegmentationConfig(target_segments=3),
            ),
        )
        assert all(
            s["embedding_kind"] == "fresh_isolated_audio_cut" for s in fresh["segments"]
        )
        for size in ("10k", "50k", "100k"):
            index = CaptionIndex(
                model,
                size,
                index_path=ROOT / f"artifacts/clap2-{model}-luna-{size}.npy",
                memory_map=True,
            )
            hits = index.retrieve(features["embeddings"][:2], 3)
            if size == "100k":
                dense = engine.index.retrieve(features["embeddings"][:2], 3)
                assert [[h["id"] for h in x] for x in hits] == [
                    [h["id"] for h in x] for x in dense
                ]
                assert np.allclose(
                    [[h["score"] for h in x] for x in hits],
                    [[h["score"] for h in x] for x in dense],
                    atol=2e-6,
                )
            del index
        if model == "m":
            (ROOT / "examples/synthetic-scene.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2)
            )
            np.savez_compressed(
                ROOT / "artifacts/example-features.npz",
                embeddings=features["embeddings"],
                bin_edges=features["bin_edges"],
            )
            from fastapi.testclient import TestClient
            from clap2_scenes.api import create_app

            with TestClient(create_app(engine)) as client:
                with (ROOT / "examples/synthetic-scene.wav").open("rb") as f:
                    response = client.post(
                        "/analyze",
                        files=[("files", ("example.wav", f, "audio/wav"))],
                        data={"top_n": "3"},
                    )
                assert response.status_code == 200, response.text
                assert (
                    len(response.json()["results"][0]["timeline"][0]["predictions"])
                    == 3
                )
        reports.append(
            {
                "model": model,
                "dimension": engine.encoder.dimension,
                "sliding_bins": len(output["timeline"]),
                "contextual_bins": len(ctx["timeline"]),
                "default_segments": len(output["segments"]),
                "fresh_segment_encoding": True,
                "all_three_indexes_verified": True,
                "mmap_gpu_top3_agreement": True,
                "wall_seconds": time.perf_counter() - start,
            }
        )
        (ROOT / "docs/model-integration-verification.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(json.dumps(reports[-1]), flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
