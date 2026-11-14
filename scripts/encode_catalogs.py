"""Encode exact candidate strings once per pinned model, then retain catalog order."""

import argparse, gzip, hashlib, json, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import open_clip
from open_clip.model import _build_text_tower
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["xxs", "xs", "s", "m", "l"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir")
    p.add_argument("--batch-size", type=int, default=1024)
    a = p.parse_args()
    torch.set_num_threads(2)
    if a.device.startswith("cuda:"):
        torch.cuda.set_device(int(a.device.split(":")[1]))
    manifest = json.loads((ROOT / "data/catalogs.json").read_text())
    registry = json.loads((ROOT / "src/clap2_scenes/models.json").read_text())
    catalogs = {}
    for key, cfg in manifest["catalogs"].items():
        with gzip.open(ROOT / cfg["file"], "rt") as f:
            catalogs[key] = [json.loads(line) for line in f]
    prompts = list(dict.fromkeys(n["text"] for rows in catalogs.values() for n in rows))
    lookup = {s: i for i, s in enumerate(prompts)}
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    for name in a.models:
        if (out / f"encoding-{name}.json").exists():
            continue
        spec = registry[name]
        folder = Path(
            snapshot_download(
                spec["repo"],
                revision=spec["revision"],
                allow_patterns=["open_clip*"],
                cache_dir=a.cache_dir,
            )
        )
        config = json.loads((folder / "open_clip_config.json").read_text())["model_cfg"]
        assert config["embed_dim"] == spec["dimension"]
        model = _build_text_tower(config["embed_dim"], config["text_cfg"])
        state = load_file(str(folder / "open_clip_model.safetensors"))
        model.load_state_dict(
            {k[5:]: v for k, v in state.items() if k.startswith("text.")}, strict=True
        )
        del state
        model = model.eval().to(a.device)
        tokenizer = open_clip.get_tokenizer("local-dir:" + str(folder))
        tokens = tokenizer(prompts, pad=False)
        lengths = np.array([len(x) for x in tokens])
        order = np.argsort(lengths, kind="stable")
        vectors = np.empty((len(prompts), spec["dimension"]), dtype="float16")
        begin = time.perf_counter()
        offset = 0

        def encode(indices, amp=True, text_model=model):
            width = min(256, ((int(lengths[indices].max()) + 7) // 8) * 8)
            x = torch.full(
                (len(indices), width), config["text_cfg"]["pad_id"], dtype=torch.long
            )
            for i, j in enumerate(indices):
                x[i, : len(tokens[j])] = tokens[j]
            with (
                torch.inference_mode(),
                torch.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=amp and a.device.startswith("cuda"),
                ),
            ):
                y = text_model(x.to(a.device))
            return F.normalize(y.float(), dim=-1).cpu().numpy()

        while offset < len(order):
            count = min(a.batch_size, len(order) - offset)
            while count > 1 and count * int(lengths[order[offset + count - 1]]) > 65536:
                count = max(1, count // 2)
            ix = order[offset : offset + count]
            vectors[ix] = encode(ix).astype("float16")
            offset += count
            if offset % 16384 < count:
                print(
                    json.dumps({"model": name, "done": offset, "total": len(order)}),
                    flush=True,
                )
        assert np.isfinite(vectors).all()
        norms = np.linalg.norm(vectors.astype("float32"), axis=1)
        assert np.max(abs(norms - 1)) < 0.001
        rng = np.random.default_rng(20261114)
        sample = rng.choice(len(prompts), 32, replace=False)
        one = np.concatenate([encode(np.array([j])) for j in sample])
        batched = vectors[sample].astype("float32")
        check = (
            (
                F.normalize(torch.from_numpy(one), dim=-1)
                * F.normalize(torch.from_numpy(batched), dim=-1)
            )
            .sum(-1)
            .numpy()
        )
        assert check.min() > 0.999
        fp32 = encode(sample, False)
        precision = (
            (
                F.normalize(torch.from_numpy(fp32), dim=-1)
                * F.normalize(torch.from_numpy(batched), dim=-1)
            )
            .sum(-1)
            .numpy()
        )
        assert precision.min() > 0.998
        report = {
            "model": name,
            **spec,
            "unique_texts": len(prompts),
            "seconds": time.perf_counter() - begin,
            "dtype": "float16",
            "precision": "BF16 inference; FP32 L2 normalization; FP16 storage",
            "max_norm_error": float(np.max(abs(norms - 1))),
            "min_single_batch_cosine": float(check.min()),
            "min_fp32_bf16_cosine": float(precision.min()),
            "max_tokens": int(lengths.max()),
            "context_limit": 256,
            "texts_reaching_context_limit": int((lengths >= 256).sum()),
            "files": {},
        }
        for key, rows in catalogs.items():
            path = out / f"clap2-{name}-luna-{key}.npy"
            np.save(
                path, vectors[[lookup[n["text"]] for n in rows]], allow_pickle=False
            )
            report["files"][key] = {
                "file": path.name,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "rows": len(rows),
                "dimension": spec["dimension"],
                "catalog_sha256": manifest["catalogs"][key]["sha256"],
            }
        (out / f"encoding-{name}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        del model, vectors
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
