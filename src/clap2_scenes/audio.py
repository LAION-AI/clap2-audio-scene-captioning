"""Bounded decoding, overlapping audio windows, and optional contextual pooling."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 32_000


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def load_audio(path: str, max_seconds: float = 600) -> tuple[np.ndarray, int]:
    """Decode a complete mono waveform at 32 kHz with ffmpeg.

    Accepted uploads are decoded through EOF, so duration does not depend on
    container duration estimates (which can truncate VBR MP3 files). For an
    overlong input, decoding stops as soon as more than ``max_seconds`` worth
    of samples arrives, then rejects the file explicitly.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source}")
    limit = _finite_float(max_seconds, "max_seconds")
    if limit <= 0:
        raise ValueError("max_seconds must be greater than zero")
    if limit > 600.0:
        raise ValueError("max_seconds cannot exceed the 600-second upload limit")
    max_samples = int(math.floor(limit * SAMPLE_RATE + 1e-9))

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    stderr_file = tempfile.TemporaryFile()
    process: subprocess.Popen[bytes] | None = None
    chunks: list[bytes] = []
    total_bytes = 0
    over_limit = False
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            bufsize=0,
        )
        assert process.stdout is not None
        while True:
            block = process.stdout.read(65_536)
            if not block:
                break
            chunks.append(block)
            total_bytes += len(block)
            # Reads may end within a sample; use floor so a partial trailing
            # sample cannot cause a false over-limit rejection.
            if total_bytes // 4 > max_samples:
                over_limit = True
                process.kill()
                break
        process.stdout.close()
        return_code = process.wait()
        if over_limit:
            raise ValueError(f"Audio exceeds the {limit:g}-second limit")
        if return_code != 0:
            stderr_file.seek(0)
            diagnostic = stderr_file.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg could not decode audio: {diagnostic or 'unknown error'}"
            )
        if total_bytes % 4:
            raise RuntimeError("ffmpeg returned an incomplete float32 audio sample")
        pcm = np.frombuffer(b"".join(chunks), dtype="<f4").astype(np.float32, copy=True)
        if pcm.size == 0:
            raise ValueError("Audio contains no decodable samples")
        if not np.isfinite(pcm).all():
            raise ValueError("Decoded audio contains non-finite samples")
        if pcm.size > max_samples:
            raise ValueError(f"Audio exceeds the {limit:g}-second limit")
        return pcm, SAMPLE_RATE
    except BaseException:
        if process is not None and process.poll() is None:
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            try:
                process.kill()
            except OSError:
                pass
            process.wait()
        raise
    finally:
        stderr_file.close()


def _time_edges(duration: float, stride: float) -> np.ndarray:
    """Make stride-spaced edges and append the exact duration as the tail."""
    full_steps = int(math.floor(duration / stride + 1e-10))
    edges = np.arange(full_steps + 1, dtype=np.float64) * stride
    if edges.size == 0:
        edges = np.array([0.0], dtype=np.float64)
    if edges[-1] >= duration - max(1e-10, duration * 1e-12):
        edges[-1] = duration
    else:
        edges = np.append(edges, duration)
    if edges.size < 2:
        edges = np.array([0.0, duration], dtype=np.float64)
    return edges


def _validate_pcm(pcm: Any) -> tuple[np.ndarray, float]:
    waveform = np.asarray(pcm, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError("pcm must be a one-dimensional mono waveform")
    if waveform.size == 0:
        raise ValueError("pcm must contain at least one sample")
    if not np.isfinite(waveform).all():
        raise ValueError("pcm contains non-finite samples")
    duration = waveform.size / SAMPLE_RATE
    if duration > 600.0:
        raise ValueError("Audio exceeds the 600-second limit")
    return np.ascontiguousarray(waveform), duration


class AudioEncoder:
    """Pinned CLAP2 audio tower. Model loading happens only on construction."""

    def __init__(self, model_size="m", device=None, batch_size=64, cache_dir=None):
        import torch, open_clip
        from huggingface_hub import snapshot_download

        registry = json.loads(Path(__file__).with_name("models.json").read_text())
        if model_size not in registry:
            raise ValueError("model_size must be xxs, xs, s, m, or l")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be between 1 and 128")
        self.spec = registry[model_size]
        self.model_size = model_size
        self.dimension = self.spec["dimension"]
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        torch.set_num_threads(2)
        if self.device.startswith("cuda:"):
            torch.cuda.set_device(int(self.device.split(":")[1]))
        folder = snapshot_download(
            self.spec["repo"],
            revision=self.spec["revision"],
            allow_patterns=["open_clip*"],
            cache_dir=cache_dir,
        )
        model, factory = open_clip.create_model_from_pretrained(
            "local-dir:" + folder, device=self.device
        )
        self.model = model.eval()
        self.preprocess = factory()
        self._torch = torch

    def _amp_context(self):
        if self.device.startswith("cuda"):
            return self._torch.autocast(
                device_type="cuda",
                dtype=(
                    self._torch.bfloat16
                    if self._torch.cuda.is_bf16_supported()
                    else self._torch.float16
                ),
            )
        return nullcontext()

    def _preprocess_windows(
        self, pcm: np.ndarray, starts: list[int], width: int
    ) -> np.ndarray:
        torch = self._torch
        prepared: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        window_count = len(starts)
        for batch_start in range(0, window_count, self.batch_size):
            batch_starts = starts[batch_start : batch_start + self.batch_size]
            prepared.clear()
            for start in batch_starts:
                end = min(start + width, pcm.size)
                crop = torch.from_numpy(pcm[start:end].copy())
                prepared.append(self.preprocess((crop, SAMPLE_RATE)))
            if not prepared:
                continue
            keys = prepared[0].keys()
            if any(item.keys() != prepared[0].keys() for item in prepared):
                raise ValueError(
                    "CLAP2 audio preprocessing returned inconsistent fields"
                )
            batch = {
                key: torch.stack([item[key] for item in prepared]).to(self.device)
                for key in keys
            }
            with torch.inference_mode(), self._amp_context():
                encoded = self.model.encode_audio(batch, normalize=True)
            encoded = encoded.float()
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
            array = encoded.cpu().numpy().astype(np.float32, copy=False)
            if array.shape != (len(batch_starts), self.dimension):
                raise ValueError(
                    f"CLAP2 returned unexpected audio embedding shape {array.shape}"
                )
            if not np.isfinite(array).all():
                raise ValueError("CLAP2 returned non-finite audio embeddings")
            vectors.append(array)
        if not vectors:
            raise ValueError("No audio windows were produced")
        return np.concatenate(vectors, axis=0)

    def _encode_sliding(
        self,
        pcm: np.ndarray,
        duration: float,
        window_seconds: float,
        stride_seconds: float,
    ):
        window = min(window_seconds, duration)
        width = max(1, int(round(window * SAMPLE_RATE)))
        hop = max(1, int(round(stride_seconds * SAMPLE_RATE)))
        if pcm.size <= width:
            starts = [0]
        else:
            last_start = pcm.size - width
            starts = list(range(0, last_start + 1, hop))
            if starts[-1] != last_start:
                starts.append(last_start)
        ends = [min(start + width, pcm.size) for start in starts]
        window_embeddings = self._preprocess_windows(pcm, starts, width)
        edges = _time_edges(duration, stride_seconds)
        output = np.zeros((edges.size - 1, self.dimension), dtype=np.float64)
        weight_sums = np.zeros(edges.size - 1, dtype=np.float64)
        win_starts_s = np.asarray(starts, dtype=np.float64) / SAMPLE_RATE
        win_ends_s = np.asarray(ends, dtype=np.float64) / SAMPLE_RATE
        for bin_i, (bin_start, bin_end) in enumerate(zip(edges[:-1], edges[1:])):
            overlap = np.maximum(
                0.0,
                np.minimum(win_ends_s, bin_end) - np.maximum(win_starts_s, bin_start),
            )
            selected = overlap > 1e-12
            if not selected.any():
                raise RuntimeError(f"No audio windows overlap time bin {bin_i}")
            weights = overlap[selected]
            output[bin_i] = (
                window_embeddings[selected].astype(np.float64) * weights[:, None]
            ).sum(axis=0)
            weight_sums[bin_i] = weights.sum()
        output /= weight_sums[:, None]
        if not np.isfinite(output).all():
            raise ValueError("Audio embedding aggregation produced non-finite values")
        return output.astype(np.float32), edges, len(starts)

    def _encode_contextual(
        self, pcm: np.ndarray, duration: float, stride_seconds: float
    ):
        if duration > 30.0:
            raise ValueError("Contextual patches support clips up to 30 seconds")
        torch = self._torch
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence

        tower = self.model.audio
        vit = tower.encoder.vit
        if (
            getattr(vit, "pool_include_prefix", False)
            or vit.attn_pool.pos_embed is not None
        ):
            raise RuntimeError(
                "CLAP2 contextual pooling requires prefix exclusion and no pool position embedding"
            )
        edges = _time_edges(duration, stride_seconds)
        waveform = torch.from_numpy(pcm.copy())
        item = self.preprocess((waveform, SAMPLE_RATE))
        if "patch_coord" not in item:
            raise RuntimeError("CLAP2 preprocessing did not provide patch_coord")
        coords = item["patch_coord"]
        if isinstance(coords, torch.Tensor):
            coords_np = coords.detach().cpu().numpy()
        else:
            coords_np = np.asarray(coords)
        if coords_np.ndim != 2 or coords_np.shape[1] < 2:
            raise RuntimeError("CLAP2 patch_coord has an unexpected shape")
        batch = {
            key: pad_sequence([value], batch_first=True).to(self.device)
            for key, value in item.items()
        }

        with torch.inference_mode(), self._amp_context():
            features = vit.forward_features(**batch)
            patches = features["patches"]
            time_index = coords_np[:, 1].astype(np.float64)
            patch_starts = time_index * 0.16
            patch_ends = np.minimum((time_index + 1.0) * 0.16, duration)
            real_durations = np.maximum(patch_ends - patch_starts, 1e-9)
            hidden = patches[
                0, vit.num_prefix_tokens : vit.num_prefix_tokens + len(coords_np)
            ]
            selected_hidden = []
            selected_priors = []
            for bin_start, bin_end in zip(edges[:-1], edges[1:]):
                overlap = np.maximum(
                    0.0,
                    np.minimum(patch_ends, bin_end)
                    - np.maximum(patch_starts, bin_start),
                )
                indices = np.flatnonzero(overlap > 1e-8)
                if indices.size == 0:
                    raise RuntimeError(
                        f"No CLAP2 audio patches overlap time bin [{bin_start:.3f}, {bin_end:.3f}]"
                    )
                selected_hidden.append(
                    hidden[torch.as_tensor(indices, device=hidden.device)]
                )
                prior = np.log(overlap[indices] / real_durations[indices])
                selected_priors.append(
                    torch.as_tensor(prior, device=hidden.device, dtype=hidden.dtype)
                )
            x = pad_sequence(selected_hidden, batch_first=True)
            mask = pad_sequence(
                selected_priors, batch_first=True, padding_value=float("-inf")
            )[:, None, None, :]
            pooled = vit.attn_pool(x, attn_mask=mask)
            pooled = vit.head(vit.head_drop(vit.fc_norm(pooled)))
            if getattr(tower, "pre_norm", False):
                pooled = F.normalize(pooled, dim=-1)
            embeddings = F.normalize(tower.proj(pooled).float(), dim=-1)
        result = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (edges.size - 1, self.dimension):
            raise ValueError(
                f"CLAP2 contextual pooling returned unexpected shape {result.shape}"
            )
        if not np.isfinite(result).all():
            raise ValueError("CLAP2 contextual pooling returned non-finite embeddings")
        return result, edges

    def encode(
        self,
        pcm: np.ndarray,
        window_seconds: float = 2.0,
        stride_seconds: float = 0.2,
        analysis_method: str = "Sliding windows",
    ) -> dict[str, Any]:
        waveform, duration = _validate_pcm(pcm)
        window = _finite_float(window_seconds, "window_seconds")
        stride = _finite_float(stride_seconds, "stride_seconds")
        if not 0.1 <= stride <= 1.0:
            raise ValueError("stride_seconds must be between 0.1 and 1.0")
        method = str(analysis_method).strip()
        if method not in {"Sliding windows", "Contextual patches"}:
            raise ValueError(
                "analysis_method must be 'Sliding windows' or 'Contextual patches'"
            )

        if method == "Sliding windows":
            if not 0.5 <= window <= 10.0:
                raise ValueError("window_seconds must be between 0.5 and 10.0")
            if stride > window:
                raise ValueError("stride_seconds cannot exceed window_seconds")
            embeddings, edges, window_count = self._encode_sliding(
                waveform, duration, window, stride
            )
        else:
            if duration > 30.0:
                raise ValueError("Contextual patches support clips up to 30 seconds")
            embeddings, edges = self._encode_contextual(waveform, duration, stride)
            window_count = 1

        return {
            "embeddings": np.ascontiguousarray(embeddings, dtype=np.float32),
            "bin_edges": np.asarray(edges, dtype=np.float64),
            "duration_seconds": float(duration),
            "window_count": int(window_count),
            "window_seconds": float(window),
            "stride_seconds": float(stride),
            "analysis_method": method,
        }
