"""Optional multipart FastAPI service. Uploaded files are temporary."""

from contextlib import asynccontextmanager
from pathlib import Path
import json, os, tempfile, threading
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from .pipeline import SceneCaptioner, AnalysisConfig


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if int(headers.get(b"content-length", b"0")) > self.max_bytes:
            await send(
                {
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"detail":"Request exceeds the upload limit"}',
                }
            )
            return
        count = 0

        async def limited():
            nonlocal count
            message = await receive()
            count += len(message.get("body", b""))
            if count > self.max_bytes:
                raise HTTPException(413, "Request exceeds the upload limit")
            return message

        await self.app(scope, limited, send)


def create_app(pipeline=None, max_upload_bytes=300_000_000):
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.captioner = pipeline or SceneCaptioner(
            model_size=os.getenv("CLAP2_MODEL", "m"),
            index_size=os.getenv("CLAP2_INDEX", "100k"),
            device=os.getenv("CLAP2_DEVICE") or None,
            retrieval_device=os.getenv("CLAP2_RETRIEVAL_DEVICE") or None,
            cache_dir=os.getenv("CLAP2_SCENES_CACHE"),
            index_path=os.getenv("CLAP2_INDEX_PATH"),
        )
        yield

    app = FastAPI(
        title="CLAP2 Audio Scene Captioning",
        version="1.0.0",
        lifespan=lifespan,
        description="Contiguous audio segmentation and exact Top-N caption retrieval. Scores are similarities, not probabilities.",
    )
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_upload_bytes)

    @app.get("/health")
    def health():
        engine = app.state.captioner
        return {
            "status": "ok",
            "model": engine.encoder.model_size,
            "index": engine.index.metadata["index_size"],
            "candidates": engine.index.count,
        }

    @app.post("/analyze")
    async def analyze(
        files: list[UploadFile] = File(...),
        top_n: int = Form(5),
        config: str = Form("{}"),
    ):
        if not 1 <= len(files) <= 8:
            raise HTTPException(422, "Upload between one and eight audio files")
        if not 1 <= top_n <= 100:
            raise HTTPException(
                422, "The HTTP service supports top_n between 1 and 100"
            )
        try:
            values = json.loads(config)
            if not isinstance(values, dict):
                raise ValueError("config must be a JSON object")
            if "top_n" in values:
                raise ValueError("Pass top_n as its own form field")
            settings = AnalysisConfig(**values, top_n=top_n)
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            with tempfile.TemporaryDirectory(prefix="clap2-scenes-") as folder:
                paths = []
                names = []
                size = 0
                for i, file in enumerate(files):
                    name = Path(file.filename or "audio").name
                    path = Path(folder) / f"{i:02d}.audio"
                    names.append(name)
                    with path.open("wb") as target:
                        while block := await file.read(1024 * 1024):
                            size += len(block)
                            if size > max_upload_bytes:
                                raise HTTPException(
                                    413, "Uploads exceed the request size limit"
                                )
                            target.write(block)
                    paths.append(path)

                def work():
                    with lock:
                        return app.state.captioner.analyze_many(paths, settings)

                results = await run_in_threadpool(work)
                for name, result in zip(names, results):
                    result["source"] = name
                return {"results": results}
        except HTTPException:
            raise
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            for file in files:
                await file.close()

    return app
