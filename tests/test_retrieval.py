import numpy as np
import pytest
from clap2_scenes.index import CaptionIndex
from clap2_scenes.segmentation import normalize


def fake_index(vectors, memory_map=False):
    i = CaptionIndex.__new__(CaptionIndex)
    i.metadata = {"dimension": vectors.shape[1]}
    i.nodes = [{"id": str(j), "text": str(j)} for j in range(len(vectors))]
    i.vectors = vectors if memory_map else normalize(vectors)
    i.memory_map = memory_map
    i.gpu_vectors = None
    return i


def test_exact_retrieval_blocks_and_ties():
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(57, 13)).astype("float16")
    vectors[7] = vectors[6]
    queries = normalize(vectors[[6, 0, 22]]) * 0.7
    a = fake_index(vectors)
    b = fake_index(vectors, True)
    scores = queries @ normalize(vectors).T
    for k in (1, 5, 57):
        expected = [
            np.lexsort((np.arange(len(vectors)), -s))[:k].tolist() for s in scores
        ]
        for index in (a, b):
            hits = index.retrieve(
                queries, k, query_batch_size=2, candidate_batch_size=7
            )
            assert [set(int(h["id"]) for h in row) for row in hits] == [
                set(row) for row in expected
            ]
            for row, ids, s in zip(hits, expected, scores):
                np.testing.assert_allclose([h["score"] for h in row], s[ids], atol=2e-7)
    assert [h["id"] for h in a.retrieve(queries[:1], 2)[0]] == ["6", "7"]
    assert abs(a.retrieve(queries[:1], 1)[0][0]["score"] - 0.7) < 1e-6
    with pytest.raises(ValueError):
        a.retrieve(queries, 58)
    with pytest.raises(ValueError):
        a.retrieve([[float("nan")] * 13])


def test_exact_score_ties_use_row_order():
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, -1], [1, 0]], dtype="float16")
    for index in (fake_index(vectors), fake_index(vectors, True)):
        hits = index.retrieve([[1, 0]], 3, candidate_batch_size=2)
        assert [h["id"] for h in hits[0]] == ["0", "2", "4"]
