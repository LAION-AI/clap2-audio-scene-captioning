"""Generate an original deterministic audio example; no evaluation recordings."""

from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sr = 32000
t = np.arange(4 * sr) / sr
rng = np.random.default_rng(17)
fade = np.minimum(1, np.minimum(t / 0.08, (4 - t) / 0.08))
a = 0.16 * (np.sin(2 * np.pi * 220 * t) + 0.35 * np.sin(2 * np.pi * 440 * t)) * fade
b = (
    0.15
    * np.sin(2 * np.pi * (800 * t + 140 * t * t))
    * (0.25 + 0.75 * (np.sin(2 * np.pi * 2 * t) > 0))
    * fade
)
c = 0.10 * rng.normal(size=len(t)) * fade
for k in range(8):
    start = int((0.2 + k * 0.45) * sr)
    n = int(0.015 * sr)
    c[start : start + n] += (
        0.5 * np.exp(-np.arange(n) / (sr * 0.003)) * rng.normal(size=n)
    )
x = np.concatenate([a, b, c])
sf.write(ROOT / "examples/synthetic-scene.wav", x, sr, subtype="PCM_16")
print("Wrote 12-second original synthetic example")
