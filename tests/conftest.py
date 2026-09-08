from __future__ import annotations

import numpy as np
import pytest

from embra import Database
from embra.util.math import l2_normalize


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "db")
    yield database
    database.close()


@pytest.fixture
def vectors(rng):
    return l2_normalize(rng.normal(size=(500, 32)).astype("float32"))


@pytest.fixture
def collection(db):
    return db.create_collection("docs", dim=32)
