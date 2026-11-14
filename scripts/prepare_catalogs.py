"""Pin and export the same 100M/M archetype captions used by the demos."""

import gzip, hashlib, json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
LIB = "laion/laion-audio-xl-caption-cluster-library"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    revision = "e4407f1f39ecc05425c5664bb68d6d9a3fb0081c"
    # Model revisions are pinned in the checked-in registry, not resolved from main.
    # The two established catalogs are read from their published, immutable files.
    catalogs = {}
    source = ROOT / "artifacts/source"
    source.mkdir(parents=True, exist_ok=True)
    for key in ["10k", "100k"]:
        name = f"indexes/subset-100m/m/luna-large{key}/nodes.json"
        p = Path(
            hf_hub_download(
                LIB, name, repo_type="dataset", revision=revision, local_dir=source
            )
        )
        doc = json.loads(p.read_text())
        rows = []
        for n in doc["nodes"]:
            row = {
                "id": n["id"],
                "text": n["prompt"],
                "origin": n["origin"],
                "path": n["path_text"],
            }
            for field in ["audio_id", "cluster_id", "centroid_cosine"]:
                if field in n:
                    row[field] = n[field]
            rows.append(row)
        catalogs[key] = (rows, doc.get("missing_caption_clusters_omitted", []), name)
    assert catalogs["10k"][0][:4416] == catalogs["100k"][0][:4416]
    luna = catalogs["10k"][0][:4416]
    assert len(luna) == 4416
    name = "subset-100m/m/k050000/representatives.parquet"
    p = hf_hub_download(
        LIB, name, repo_type="dataset", revision=revision, local_dir=source
    )
    table = pq.read_table(
        p,
        columns=[
            "cluster_id",
            "rank",
            "audio_id",
            "short_caption",
            "cosine_similarity",
        ],
    )
    table = table.filter(pc.equal(table["rank"], 1))
    rows = list(luna)
    omitted = []
    for n in table.to_pylist():
        if not n["short_caption"] or not n["short_caption"].strip():
            omitted.append(n["cluster_id"])
            continue
        rows.append(
            {
                "id": f"M100M_K50000_{n['cluster_id']}",
                "text": n["short_caption"],
                "origin": "cluster_archetype",
                "path": f"M · 100M captions · 50,000 clusters · cluster {n['cluster_id']:,} · audio {n['audio_id']}",
                "audio_id": n["audio_id"],
                "cluster_id": n["cluster_id"],
                "centroid_cosine": n["cosine_similarity"],
            }
        )
    catalogs["50k"] = (rows, omitted, name)
    manifest = {
        "version": "v1.0.0",
        "source_repository": LIB,
        "source_revision": revision,
        "training_model": "m",
        "training_rows": 100000000,
        "luna_nodes": 4416,
        "catalogs": {},
    }
    for key in ["10k", "50k", "100k"]:
        rows, omitted, name = catalogs[key]
        assert len(set(x["id"] for x in rows)) == len(rows)
        raw = "".join(
            json.dumps(n, ensure_ascii=False, separators=(",", ":")) + "\n"
            for n in rows
        ).encode()
        file = ROOT / f"data/candidates/luna-{key}.jsonl.gz"
        file.write_bytes(gzip.compress(raw, mtime=0))
        manifest["catalogs"][key] = {
            "rows": len(rows),
            "archetypes": len(rows) - 4416,
            "requested_clusters": int(key[:-1]) * 1000,
            "missing_caption_clusters_omitted": omitted,
            "file": str(file.relative_to(ROOT)),
            "sha256": sha(file),
            "ordered_ids_sha256": hashlib.sha256(
                "\n".join(x["id"] for x in rows).encode()
            ).hexdigest(),
            "source_file": name,
        }
    (ROOT / "data/catalogs.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
