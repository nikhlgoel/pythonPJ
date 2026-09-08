import numpy as np
import pytest

from embra.errors import SchemaError
from embra.types import Metric
from embra.util.math import coerce_vectors, distance_matrix, l2_normalize, validate_vector


def test_l2_normalize_is_unit_norm(rng):
    x = rng.normal(size=(10, 5))
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0)


def test_l2_normalize_handles_zero_vector():
    out = l2_normalize(np.zeros((1, 4)))
    assert np.all(np.isfinite(out))


def test_validate_vector_rejects_wrong_dim():
    with pytest.raises(SchemaError):
        validate_vector([1, 2, 3], dim=4)


def test_validate_vector_rejects_nan():
    with pytest.raises(SchemaError):
        validate_vector([1.0, float("nan")], dim=2)


def test_coerce_vectors_promotes_1d():
    assert coerce_vectors([1, 2, 3], dim=3).shape == (1, 3)


@pytest.mark.parametrize("metric", list(Metric))
def test_distance_matrix_matches_bruteforce(rng, metric):
    data = l2_normalize(rng.normal(size=(20, 6))).astype("float32")
    q = l2_normalize(rng.normal(size=6)).astype("float32")
    got = distance_matrix(q, data, metric)
    if metric is Metric.L2:
        want = np.sum((data - q) ** 2, axis=1)
    elif metric is Metric.COSINE:
        want = 1 - data @ q
    else:
        want = -(data @ q)
    assert np.allclose(got, want, atol=1e-5)


def test_distance_matrix_empty():
    assert distance_matrix(np.zeros(3, "float32"), np.zeros((0, 3), "float32"), Metric.L2).size == 0
