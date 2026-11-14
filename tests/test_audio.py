import numpy as np
import pytest
from clap2_scenes.audio import AudioEncoder, _time_edges, load_audio
from clap2_scenes.smoothing import smooth_scores
from clap2_scenes.segmentation import normalize


def test_weighting_equals_mean_cosine_and_tail():
    encoder = AudioEncoder.__new__(AudioEncoder)
    encoder.dimension = 4

    def windows(pcm, starts, width):
        return normalize(np.array([[1, i + 1, i % 3, 1] for i in range(len(starts))]))

    encoder._preprocess_windows = windows
    pcm = np.zeros(int(3.13 * 32000), "float32")
    rows, edges, n = encoder._encode_sliding(pcm, 3.13, 2, 0.2)
    assert edges[-1] == 3.13 and len(rows) == 16 and n == 7
    starts = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.13]
    w = windows(pcm, starts, 64000)
    text = normalize([[0.5, 0.8, 0.7, 0.2]])[0]
    for row, a, b in zip(rows, edges[:-1], edges[1:]):
        weights = np.maximum(
            0, np.minimum(np.array(starts) + 2, b) - np.maximum(starts, a)
        )
        assert abs(row @ text - np.average(w @ text, weights=weights)) < 2e-7
    assert np.any(np.linalg.norm(rows, axis=1) < 0.999)


@pytest.mark.parametrize("method", ["Moving average", "Gaussian"])
def test_feature_smoothing_equals_score_smoothing(method):
    rng = np.random.default_rng(2)
    x = rng.normal(size=(51, 7)).astype("float32")
    t = normalize(rng.normal(size=(17, 7)))
    edges = np.r_[np.arange(51) * 0.2, 10.13]
    np.testing.assert_allclose(
        smooth_scores(x, edges, method, 1) @ t.T,
        smooth_scores(x @ t.T, edges, method, 1),
        atol=7e-7,
    )


def test_duration_rejected_without_truncation(tmp_path):
    import soundfile as sf

    path = tmp_path / "audio.wav"
    sf.write(path, np.zeros(32001), 32000)
    with pytest.raises(ValueError, match="exceeds"):
        load_audio(str(path), max_seconds=1)
    pcm, sr = load_audio(str(path))
    assert len(pcm) == 32001 and sr == 32000
    assert _time_edges(0.01, 0.2).tolist() == [0, 0.01]
