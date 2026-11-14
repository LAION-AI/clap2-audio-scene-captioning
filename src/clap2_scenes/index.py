"""Versioned text catalogs, checksum-verified downloads and exact bounded retrieval."""

import gzip, hashlib, json, os, tempfile, urllib.request
from pathlib import Path
import numpy as np
from .segmentation import normalize


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, path, expected_sha256):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected_sha256:
        return path
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "clap2-scenes/1.0"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as source,
            open(tmp, "wb") as target,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
        if sha256(tmp) != expected_sha256:
            raise ValueError(
                "Downloaded index checksum does not match the release manifest"
            )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


class CaptionIndex:
    def __init__(
        self,
        model_size="m",
        index_size="100k",
        cache_dir=None,
        device="cpu",
        memory_map=False,
        index_path=None,
    ):
        root = Path(__file__).parent
        manifest = json.loads((root / "indexes.json").read_text())
        if (
            model_size not in manifest["models"]
            or index_size not in manifest["catalogs"]
        ):
            raise ValueError("Unknown model or index size")
        self.metadata = {
            **manifest["models"][model_size],
            **manifest["catalogs"][index_size],
            "model_size": model_size,
            "index_size": index_size,
        }
        spec = manifest["embeddings"][model_size][index_size]
        self.metadata["embedding_sha256"] = spec["sha256"]
        self.spec = spec
        catalog = root / "data" / Path(self.metadata["file"]).name
        if sha256(catalog) != self.metadata["sha256"]:
            raise ValueError("Candidate text checksum mismatch")
        with gzip.open(catalog, "rt", encoding="utf8") as f:
            self.nodes = [json.loads(line) for line in f]
        if len(self.nodes) != spec["rows"] or len({n["id"] for n in self.nodes}) != len(
            self.nodes
        ):
            raise ValueError("Invalid candidate IDs or row count")
        ids_hash = hashlib.sha256(
            "\n".join(n["id"] for n in self.nodes).encode()
        ).hexdigest()
        if ids_hash != self.metadata["ordered_ids_sha256"]:
            raise ValueError("Candidate row order does not match the index")
        if index_path:
            path = Path(index_path)
            if sha256(path) != spec["sha256"]:
                raise ValueError("Local index checksum mismatch")
        else:
            directory = Path(
                cache_dir
                or os.environ.get(
                    "CLAP2_SCENES_CACHE", Path.home() / ".cache/clap2-scenes"
                )
            )
            path = download(spec["url"], directory / spec["file"], spec["sha256"])
        self.vectors = np.load(
            path, mmap_mode="r" if memory_map else None, allow_pickle=False
        )
        if (
            self.vectors.shape != (len(self.nodes), self.metadata["dimension"])
            or self.vectors.dtype != np.float16
        ):
            raise ValueError("Index shape or dtype mismatch")
        self.memory_map = memory_map
        self.device = str(device)
        self.gpu_vectors = None
        if not memory_map:
            self.vectors = normalize(self.vectors)
            if self.device.startswith("cuda"):
                import torch

                torch.backends.cuda.matmul.allow_tf32 = False
                self.gpu_vectors = torch.from_numpy(self.vectors).to(self.device)
                self.vectors = None
        elif self.device != "cpu":
            raise ValueError('Memory mapping uses CPU retrieval; choose device="cpu"')

    @property
    def count(self):
        return len(self.nodes)

    def retrieve(
        self, queries, top_n=5, query_batch_size=32, candidate_batch_size=8192
    ):
        q = np.asarray(queries, dtype="float32")
        if (
            q.ndim != 2
            or q.shape[1] != self.metadata["dimension"]
            or not np.isfinite(q).all()
        ):
            raise ValueError("Queries must be finite and match the index dimension")
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or not 1 <= top_n <= self.count
        ):
            raise ValueError(
                "top_n must be an integer between 1 and the candidate count"
            )
        if query_batch_size < 1 or candidate_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        output = []

        def best(scores, ids):
            result = []
            for row in scores:
                k = min(top_n, len(ids))
                threshold = np.partition(row, len(row) - k)[len(row) - k]
                selected = np.flatnonzero(row >= threshold)
                ordered = selected[np.lexsort((ids[selected], -row[selected]))[:k]]
                result.append(ordered)
            return result

        for start in range(0, len(q), query_batch_size):
            x = q[start : start + query_batch_size]
            if self.gpu_vectors is not None:
                import torch

                with torch.inference_mode():
                    scores = (
                        (torch.from_numpy(x).to(self.device) @ self.gpu_vectors.T)
                        .cpu()
                        .numpy()
                    )
                ids = np.arange(self.count)
                chosen = best(scores, ids)
            elif not self.memory_map:
                scores = x @ self.vectors.T
                ids = np.arange(self.count)
                chosen = best(scores, ids)
            else:
                running_scores = np.empty((len(x), 0), dtype="float32")
                running_ids = np.empty((len(x), 0), dtype="int64")
                for begin in range(0, self.count, candidate_batch_size):
                    v = normalize(self.vectors[begin : begin + candidate_batch_size])
                    block = x @ v.T
                    joined = np.concatenate([running_scores, block], axis=1)
                    joined_ids = np.concatenate(
                        [
                            running_ids,
                            np.broadcast_to(
                                np.arange(begin, begin + len(v)), block.shape
                            ),
                        ],
                        axis=1,
                    )
                    next_scores = []
                    next_ids = []
                    for row, identifiers in zip(joined, joined_ids):
                        pick = best(row[None], identifiers)[0]
                        next_scores.append(row[pick])
                        next_ids.append(identifiers[pick])
                    running_scores = np.stack(next_scores)
                    running_ids = np.stack(next_ids)
                for row, identifiers in zip(running_scores, running_ids):
                    output.append(
                        [
                            {**self.nodes[int(i)], "score": float(s)}
                            for i, s in zip(identifiers, row)
                        ]
                    )
                continue
            for row, pick in zip(scores, chosen):
                output.append(
                    [{**self.nodes[int(i)], "score": float(row[i])} for i in pick]
                )
        return output
