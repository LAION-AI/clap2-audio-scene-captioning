"""Draw the README figure from an actual saved prediction result."""

from pathlib import Path
import json, textwrap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "examples/synthetic-scene.json").read_text())
y, sr = sf.read(ROOT / "examples/synthetic-scene.wav")
colors = ["#64b7a8", "#8cb6e1", "#b69ad6", "#dda965", "#d18fa0"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#d0dcdf",
        "text.color": "#1c343e",
        "axes.labelcolor": "#526770",
        "xtick.color": "#526770",
        "ytick.color": "#526770",
    }
)
fig = plt.figure(figsize=(12, 6.2), facecolor="#f3f7f8")
gs = fig.add_gridspec(
    3,
    1,
    height_ratios=[1.5, 0.48, 2.6],
    hspace=0.4,
    left=0.05,
    right=0.98,
    top=0.83,
    bottom=0.08,
)
fig.text(0.05, 0.93, "From sound to scenes", fontsize=25, weight="bold")
fig.text(
    0.05,
    0.875,
    "CLAP2 M  ·  104,401 candidates  ·  2 s windows / 0.2 s stride  ·  actual model output",
    fontsize=11,
    color="#526770",
)
ax = fig.add_subplot(gs[0])
ax.set_facecolor("#f3f7f8")
ax.plot(np.arange(0, len(y), 80) / sr, y[::80], color="#257e80", lw=0.65)
ax.set_xlim(0, 12)
ax.set_yticks([])
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
ax.set_xticklabels(["0 s", "2 s", "4 s", "6 s", "8 s", "10 s", "12 s"])
ax.axvline(4, color="#789198", ls=":", lw=1)
ax.axvline(8, color="#789198", ls=":", lw=1)
for t, label in [
    (2, "Constructed tone"),
    (6, "Synthetic chirps"),
    (10, "Noise + clicks"),
]:
    ax.text(t, 0.45, label, ha="center", fontsize=9, color="#526770")
bar = fig.add_subplot(gs[1])
bar.set_xlim(0, 12)
bar.set_ylim(0, 1)
bar.axis("off")
for i, s in enumerate(r["segments"]):
    bar.barh(
        0.5,
        s["end"] - s["start"],
        left=s["start"],
        height=0.85,
        color=colors[i % 5],
        edgecolor="#f3f7f8",
        linewidth=2,
    )
    bar.text(
        (s["start"] + s["end"]) / 2,
        0.5,
        str(i + 1),
        ha="center",
        va="center",
        weight="bold",
        color="#19333a",
    )
textax = fig.add_subplot(gs[2])
textax.axis("off")
for i, s in enumerate(r["segments"]):
    y0 = 0.95 - i * 0.19
    hit = s["predictions"][0]
    textax.text(
        0, y0, f"{i + 1:02d}", weight="bold", color=colors[i % 5], fontsize=13, va="top"
    )
    textax.text(0.045, y0, f"{s['start']:.1f}–{s['end']:.1f} s", fontsize=10, va="top")
    textax.text(
        0.15, y0, "\n".join(textwrap.wrap(hit["text"], 108)), fontsize=10, va="top"
    )
    textax.text(
        0.99,
        y0,
        f"{hit['score']:.3f}",
        ha="right",
        va="top",
        fontsize=10,
        weight="bold",
    )
fig.text(
    0.05,
    0.025,
    "Retrieved captions are hypotheses, not ground truth. Scores are cosine similarities, not probabilities.",
    fontsize=9,
    color="#526770",
)
fig.savefig(ROOT / "docs/assets/example-timeline.svg", facecolor=fig.get_facecolor())
fig.savefig(
    ROOT / "artifacts/example-timeline.png", dpi=140, facecolor=fig.get_facecolor()
)
