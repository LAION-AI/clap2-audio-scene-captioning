# Functional verification

Checks were run on 2026-11-14 with Python 3.13, PyTorch 2.9.1+cu128, OpenCLIP 4.2.2 and an NVIDIA A100 80 GB. No caption-accuracy benchmark is claimed.

## Automated tests

Run `pytest -q` after installing the development and API extras. **26 tests passed.** They cover:

- All five Python segmentation methods compared to fixed outputs from the existing Space's JavaScript implementation, including tuned local statistics, feature smoothing and peak settings.
- Continuous coverage, exact tail handling, minimum-duration merging, short recordings, constant features and invalid inputs.
- Equality of weighted feature dot products to mean window/text cosines, and equivalence of feature smoothing to score smoothing.
- Resident versus blockwise exact retrieval, Top-N bounds and exact-score tie ordering.
- Checksum verification, repairing a corrupt cached download, and rejection/cleanup of mismatched downloads.
- Reusing cached features, rejecting incompatible timing settings, ordered multipart batches, API validation, upload size limits (with and without Content-Length) and temporary-file cleanup.

The test fixture stores Space reference outputs directly; tests do not depend on a local Space checkout or Node.js.

## Real checkpoints and matrices

[`model-integration-verification.json`](model-integration-verification.json) records an actual run for **XXS, XS, S, M and L**. For each model:

- The complete synthetic example was encoded with sliding windows and contextual patches, producing 60 bins in each mode.
- Default segmentation covered the full clip and respected the 1-second minimum.
- Smoothing and Top-N were changed using cached features.
- Isolated segments were freshly encoded and retrieved.
- All three candidate matrices were loaded and checked, covering all **15 assets**.
- CPU memory-mapped and GPU resident Top-3 IDs agreed on the checked queries, with scores within numerical tolerance.

The M run also uploaded the example to the actual FastAPI application using its test client and checked its returned Top-3 predictions. The reported per-model wall times include several analyses and index checks; they are **not** single-inference benchmarks.

[`encoding-verification.json`](encoding-verification.json) records all text tower runs, full-file hashes, finite/unit-norm checks, maximum token length and 32-prompt singleton/batch plus FP32/BF16 comparisons per model.

## Browser

The exported HTML report was tested in headless Chromium at desktop and mobile widths. Playback advanced, seeking updated both panels, colored segment buttons selected the correct segment, and JSON export produced a download. There were no browser errors or mobile horizontal overflow. See [`browser-verification.json`](browser-verification.json) and the [desktop screenshot](assets/report-desktop.png).

The report contains the original generated example converted to mono MP3 at 128 kbit/s with `preload="none"`. It requires no CDN JavaScript, network caption service or external font.

## Scope

The audio example is synthetic and deliberately simple. Passing these checks establishes that the software and artifacts interoperate; it does not establish semantic caption accuracy on real-world recordings, probability calibration, or event-boundary quality. GPU/CPU kernels and near-tied scores may produce small numerical differences on other systems.

## Ten-minute recording

The complete 600-second synthetic test produced **3,000 time bins**, **2,991 overlapping windows**, five segments covering the full duration, and Top-5 matches in every bin. On the A100 test run, after model/index loading, audio decoding and encoding took **5.99 s** and retrieval plus segmentation took **1.54 s**, for **7.53 s total**. This is one measured synthetic workload, not a general latency guarantee. See [`long-audio-verification.json`](long-audio-verification.json).

## Public release and installed package

All **15 public GitHub release matrices** were downloaded anonymously and hashed end to end; sizes and SHA-256 values match the manifest. See [`public-release-verification.json`](public-release-verification.json).

The built wheel was installed into a separate directory and run outside the checkout. It successfully downloaded its index without authentication, ran XXS CPU inference on one file and an ordered two-file batch, served API documentation and health metadata, and rejected invalid audio. See [`installed-package-verification.json`](installed-package-verification.json).
