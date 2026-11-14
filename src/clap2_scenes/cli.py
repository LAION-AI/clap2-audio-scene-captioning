"""Command-line annotation, index download, report export and HTTP serving."""

import argparse, json, os
from pathlib import Path
from .pipeline import SceneCaptioner, AnalysisConfig
from .segmentation import SegmentationConfig


def main():
    p = argparse.ArgumentParser(prog="clap2-scenes")
    sub = p.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Annotate one or more local audio files")
    analyze.add_argument("audio", nargs="+")
    analyze.add_argument("-o", "--output", required=True)
    for name, default in [
        ("model", "m"),
        ("index", "100k"),
        ("device", None),
        ("retrieval-device", None),
        ("cache-dir", None),
        ("index-path", None),
    ]:
        analyze.add_argument("--" + name, default=default)
    analyze.add_argument("--window", type=float, default=2)
    analyze.add_argument("--stride", type=float, default=0.2)
    analyze.add_argument("--top-n", type=int, default=5)
    analyze.add_argument("--segments", type=int, default=5)
    analyze.add_argument("--min-segment", type=float, default=1)
    analyze.add_argument(
        "--method",
        choices=["hierarchical", "adaptive", "fixed", "percentile", "novelty"],
        default="hierarchical",
    )
    analyze.add_argument(
        "--features", choices=["sliding", "contextual"], default="sliding"
    )
    analyze.add_argument(
        "--smoothing",
        choices=["none", "moving_average", "gaussian"],
        default="moving_average",
    )
    analyze.add_argument("--smoothing-width", type=float, default=1)
    analyze.add_argument("--config", help="JSON file with the complete AnalysisConfig")
    analyze.add_argument("--memory-map", action="store_true")
    analyze.add_argument("--include-embeddings", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", default="m")
    serve.add_argument("--index", default="100k")
    serve.add_argument("--device")
    serve.add_argument("--index-path")
    download = sub.add_parser("download")
    download.add_argument("--model", default="m")
    download.add_argument("--index", default="100k")
    download.add_argument("--cache-dir")
    render = sub.add_parser("render")
    render.add_argument("result")
    render.add_argument("--audio", required=True)
    render.add_argument("-o", "--output", required=True)
    a = p.parse_args()
    if a.command == "serve":
        import uvicorn
        from .api import create_app

        for key, value in [
            ("CLAP2_MODEL", a.model),
            ("CLAP2_INDEX", a.index),
            ("CLAP2_DEVICE", a.device),
            ("CLAP2_INDEX_PATH", a.index_path),
        ]:
            if value is not None:
                os.environ[key] = value
        uvicorn.run(create_app(), host=a.host, port=a.port)
        return
    if a.command == "download":
        from .index import download as fetch

        manifest = json.loads(Path(__file__).with_name("indexes.json").read_text())
        spec = manifest["embeddings"][a.model][a.index]
        directory = Path(
            a.cache_dir
            or os.environ.get("CLAP2_SCENES_CACHE", Path.home() / ".cache/clap2-scenes")
        )
        print(fetch(spec["url"], directory / spec["file"], spec["sha256"]))
        return
    if a.command == "render":
        from .report import render_report

        result = json.loads(Path(a.result).read_text())
        result = result.get("results", result) if isinstance(result, dict) else result
        result = result[0] if isinstance(result, list) else result
        render_report(result, a.audio, a.output)
        print(a.output)
        return
    config = (
        AnalysisConfig(**json.loads(Path(a.config).read_text()))
        if a.config
        else AnalysisConfig(
            window_seconds=a.window,
            stride_seconds=a.stride,
            top_n=a.top_n,
            feature_method=a.features,
            smoothing=a.smoothing,
            smoothing_seconds=a.smoothing_width,
            include_embeddings=a.include_embeddings,
            segmentation=SegmentationConfig(
                method=a.method,
                target_segments=a.segments,
                min_segment_seconds=a.min_segment,
            ),
        )
    )
    engine = SceneCaptioner(
        a.model,
        a.index,
        a.device,
        a.retrieval_device,
        a.cache_dir,
        memory_map=a.memory_map,
        index_path=a.index_path,
    )
    results = engine.analyze_many(a.audio, config)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(a.output)


if __name__ == "__main__":
    main()
