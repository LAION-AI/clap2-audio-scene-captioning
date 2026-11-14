# Candidate catalogs and text embedding indexes

## What is in an index?

Every index pairs one ordered text catalog with a dense NumPy matrix. Row `i` is the embedding of catalog row `i`'s exact `text` string.

The three catalogs are in [`data/candidates/`](../data/candidates/) as UTF-8 JSON Lines compressed with gzip. They are also included as Python package resources, so an installed wheel can use them without a Git checkout. The manifest [`data/indexes.json`](../data/indexes.json) records catalog hashes, ordered-ID hashes, model revisions, dimensions, matrix hashes, byte sizes, URLs, selection provenance and omitted cluster IDs.

The matrices are **GitHub release assets**, not files stored in Git history:

```text
clap2-{xxs,xs,s,m,l}-luna-{10k,50k,100k}.npy
```

Each is a contiguous row-major `float16` matrix, generated from L2-normalized text embeddings. Loading renormalizes vectors in float32 to correct FP16 storage rounding. `allow_pickle=False` is used. These matrices contain caption text embeddings, not cluster centroids or audio features.

| Catalog | Rows | Source |
|---|---:|---|
| `10k` | 14,416 | 4,416 Luna rows + 10,000 M/100M rank-1 captions |
| `50k` | 54,410 | 4,416 Luna rows + 49,994 M/100M rank-1 captions |
| `100k` | 104,401 | 4,416 Luna rows + 99,985 M/100M rank-1 captions |

Candidate text can repeat under distinct IDs, including across clusters. Such rows are preserved. Encoding deduplicates exact strings only as a computation optimization, then restores all original rows and IDs. There are **142,396 distinct strings across the three catalogs**. All five model sizes encode this same union.

Luna rows preserve taxonomy paths, including intermediate and leaf nodes. They use `origin="luna_taxonomy"`; archetypes use `origin="cluster_archetype"` and preserve `audio_id`, `cluster_id`, and `centroid_cosine`. The latter is selection-time metadata from the source clustering; it is not an audio matching score.

## Get the binaries

The [v1.0.0 release](https://github.com/LAION-AI/clap2-audio-scene-captioning/releases/tag/v1.0.0) contains all 15 matrices and `SHA256SUMS`.

```bash
# Download one matrix without loading a model.
clap2-scenes download --model m --index 100k
```

Or download the corresponding release asset manually and pass `index_path`:

```python
from clap2_scenes import SceneCaptioner
engine = SceneCaptioner(
    model_size="xs",
    index_size="50k",
    index_path="/data/clap2-xs-luna-50k.npy",
    model_cache_dir="/data/huggingface-cache",
)
```

The library validates the complete matrix SHA-256, catalog hash, ordered IDs, shape, dtype and model revision. Downloads use a temporary file and an atomic rename only after checksum verification. The exact pinned Hugging Face checkpoint must also be cached for offline model loading; use `HF_HUB_OFFLINE=1` after downloading it once.

## RAM, GPU and memory mapping

By default, retrieval follows the audio encoder's device. GPU retrieval stores normalized float32 text vectors on that GPU. CPU retrieval keeps normalized float32 vectors in RAM. `retrieval_device="cpu"` lets the audio encoder use CUDA while the index stays in host RAM.

```python
# CPU index backed by a memory-mapped FP16 file; audio still runs on CUDA.
engine = SceneCaptioner(
    model_size="m",
    index_size="100k",
    device="cuda",
    retrieval_device="cpu",
    memory_map=True,
)
```

Memory mapping scans 8,192 candidates at a time, normalizes each block and merges exact Top-N results. The operating system may still cache file pages in RAM. It reduces explicit resident vector allocation, not the amount of work required for exact search. On a cold SSD cache, I/O can dominate. No FAISS service or approximate-search training is required for these candidate counts.

Search batches at most 32 query rows by default. Resident mode therefore holds a temporary score matrix of at most `32 × candidate_count × 4` bytes, about 12.7 MiB for the largest catalog. Memory-mapped mode instead uses bounded candidate blocks plus the running Top-N. Top-N ties among exactly equal computed scores use the earlier catalog row; tiny floating-point differences across devices or block shapes can alter very close ranks.

For dimensions `D` and candidates `K`, vector storage is `2KD` bytes on disk and approximately `4KD` bytes resident. Work scales approximately with `query_count × K × D`; latency also depends on device, batching, disk cache, and Top-N. The encoder and decoded PCM are additional memory consumers.

## Recreate the release

[`prepare_catalogs.py`](../scripts/prepare_catalogs.py) reads these **pinned** source files at revision `e4407f1f39ecc05425c5664bb68d6d9a3fb0081c` of [the caption cluster library](https://huggingface.co/datasets/laion/laion-audio-xl-caption-cluster-library/tree/e4407f1f39ecc05425c5664bb68d6d9a3fb0081c):

- `indexes/subset-100m/m/luna-large10k/nodes.json`
- `indexes/subset-100m/m/luna-large100k/nodes.json`
- `subset-100m/m/k050000/representatives.parquet`, selecting `rank == 1`

The source subset consists of 100M valid short-caption embeddings selected from deterministically shuffled training shards, rather than a uniform random sample of every row. Selection details and upstream E-means code are in the source library. This repository reuses its completed selection; it does not rerun the 100M clustering.

[`encode_catalogs.py`](../scripts/encode_catalogs.py) loads only each checkpoint's text tower, uses length-sorted batches of up to 1,024 texts with a 65,536 padded-token budget, and encodes in BF16 on the tested A100 hardware. It normalizes in FP32 and stores FP16. All texts fit within the 256-token context; the longest contained 146 tokens. Singleton/batch and FP32/BF16 consistency checks are recorded in [`encoding-verification.json`](encoding-verification.json).

[`finalize_indexes.py`](../scripts/finalize_indexes.py) copies catalogs into the package and builds the manifests and checksums. It assumes all five encoding reports exist. Release URLs target v1.0.0; change them when publishing a new version. Floating-point kernels can vary, so a fresh run need not reproduce the binary hashes bit for bit.

## Provenance and licenses

- **Code:** Apache-2.0; see [LICENSE](../LICENSE).
- **CLAP2 checkpoints:** official [LAION CLAP2 model repositories](https://huggingface.co/laion/models?search=clap2), linked individually with immutable revisions in [`models.json`](../src/clap2_scenes/models.json). Their model cards specify Apache-2.0.
- **Cluster captions and identifiers:** [LAION-Audio-XL caption cluster library](https://huggingface.co/datasets/laion/laion-audio-xl-caption-cluster-library), derived from [LAION-Audio-XL](https://huggingface.co/datasets/laion/laion-audio-xl) and its [embedding release](https://huggingface.co/datasets/laion/laion-audio-xl-embeddings). They retain upstream provenance and applicable terms; this repository does not grant rights to the source recordings.
- **Taxonomy attribution:** [laion/audio-taxonomy-classifier](https://huggingface.co/laion/audio-taxonomy-classifier), revision `dd45763a1988f8c6cbba37e88a95ab5bd7053366`, and its authors; vocal-burst classes originate in [LAION-AI/voice-taxonomies](https://github.com/LAION-AI/voice-taxonomies) under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Luna wording is a paraphrased adaptation.
- **Luna taxonomy sentences:** reused verbatim from the pinned cluster-library indexes. They are short paraphrases of the CLAP2 / LAION-Audio-XL taxonomy, including named intermediate nodes. They are not newly generated or independently human-verified in this repository.
- **Synthetic example audio:** generated locally by [`make_example.py`](../scripts/make_example.py) and distributed under Apache-2.0 with the code.

No source audio from LAION-Audio-XL is redistributed. The source catalog does not provide a single blanket license for every underlying recording; consult the linked dataset and model cards for the source materials you use. Preserve IDs and attribution when redistributing derived catalogs.
