"""Create the portable release manifest and package the exact candidate strings."""

import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = ROOT / "src/clap2_scenes"
catalogs = json.loads((ROOT / "data/catalogs.json").read_text())
models = json.loads((pkg / "models.json").read_text())
manifest = {**catalogs, "models": models, "embeddings": {}}
(pkg / "data").mkdir(exist_ok=True)
for key, cfg in manifest["catalogs"].items():
    shutil.copy2(ROOT / cfg["file"], pkg / "data" / Path(cfg["file"]).name)
    cfg.update(
        training_rows=catalogs["training_rows"],
        selection_model="m",
        source_repository=catalogs["source_repository"],
        source_revision=catalogs["source_revision"],
    )
reports = {}
for name in models:
    r = json.loads((ROOT / f"artifacts/encoding-{name}.json").read_text())
    reports[name] = r
    manifest["embeddings"][name] = {}
    for key, spec in r["files"].items():
        assert spec["catalog_sha256"] == manifest["catalogs"][key]["sha256"]
        manifest["embeddings"][name][key] = {
            **spec,
            "url": "https://github.com/LAION-AI/clap2-audio-scene-captioning/releases/download/v1.0.0/"
            + spec["file"],
        }
(pkg / "indexes.json").write_text(json.dumps(manifest, indent=2))
(ROOT / "data/indexes.json").write_text(json.dumps(manifest, indent=2))
(ROOT / "docs/encoding-verification.json").write_text(json.dumps(reports, indent=2))
checks = [
    f"{s['sha256']}  {s['file']}"
    for variants in manifest["embeddings"].values()
    for s in variants.values()
]
(ROOT / "artifacts/SHA256SUMS").write_text("\n".join(checks) + "\n")
print(
    "Prepared",
    len(checks),
    "embedding assets, bytes",
    sum(s["bytes"] for v in manifest["embeddings"].values() for s in v.values()),
)
