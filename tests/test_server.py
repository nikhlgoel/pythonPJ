import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from annex import Database  # noqa: E402
from annex.server.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app = create_app(Database(tmp_path / "srv"))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seeded(client):
    client.post("/v1/collections", json={"name": "docs", "dim": 4})
    rng = np.random.default_rng(0)
    docs = [
        {
            "id": f"d{i}",
            "vector": rng.normal(size=4).tolist(),
            "metadata": {"group": i % 3},
            "text": f"document {i} about vectors",
        }
        for i in range(30)
    ]
    client.post("/v1/collections/docs/documents", json={"documents": docs})
    return client, docs


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "version" in body


def test_create_and_list_collections(client):
    assert client.post("/v1/collections", json={"name": "c", "dim": 8}).status_code == 201
    assert client.get("/v1/collections").json()["collections"] == ["c"]
    assert client.get("/v1/collections/c").json()["config"]["dim"] == 8


def test_create_collection_validates_input(client):
    assert client.post("/v1/collections", json={"name": "c", "dim": 0}).status_code == 422
    assert client.post("/v1/collections", json={"name": "c", "dim": 4, "metric": "x"}).status_code == 422


def test_missing_collection_is_404(client):
    assert client.get("/v1/collections/ghost").status_code == 404
    assert client.post("/v1/collections/ghost/search", json={"vector": [1, 2]}).status_code == 404


def test_upsert_and_get_document(seeded):
    client, _ = seeded
    body = client.get("/v1/collections/docs/documents/d3?include_vector=true").json()
    assert body["id"] == "d3" and len(body["vector"]) == 4
    assert client.get("/v1/collections/docs/documents/nope").status_code == 404


def test_search_returns_hits_and_plan(seeded):
    client, docs = seeded
    res = client.post(
        "/v1/collections/docs/search", json={"vector": docs[5]["vector"], "k": 3}
    ).json()
    assert res["hits"][0]["id"] == "d5"
    assert res["plan"]["strategy"] in ("exact_scan", "ann_graph")
    assert res["took_ms"] >= 0


def test_search_with_filter_and_text(seeded):
    client, docs = seeded
    res = client.post(
        "/v1/collections/docs/search",
        json={"vector": docs[0]["vector"], "text": "document 0", "k": 5, "filter": {"group": 0}},
    ).json()
    assert all(h["metadata"]["group"] == 0 for h in res["hits"])


def test_search_requires_a_query(seeded):
    client, _ = seeded
    assert client.post("/v1/collections/docs/search", json={"k": 3}).status_code == 400


def test_search_validates_k(seeded):
    client, docs = seeded
    body = {"vector": docs[0]["vector"], "k": 0}
    assert client.post("/v1/collections/docs/search", json=body).status_code == 422


def test_delete_document(seeded):
    client, _ = seeded
    assert client.delete("/v1/collections/docs/documents/d1").status_code == 204
    assert client.get("/v1/collections/docs/documents/d1").status_code == 404
    assert client.delete("/v1/collections/docs/documents/d1").status_code == 404


def test_maintenance_endpoints(seeded):
    client, _ = seeded
    assert "reclaimed" in client.post("/v1/collections/docs/vacuum").json()
    assert client.post("/v1/collections/docs/flush").json()["records"] == 30
    assert client.post("/v1/collections/docs/rebuild").json()["documents"] == 30


def test_drop_collection(seeded):
    client, _ = seeded
    assert client.delete("/v1/collections/docs").status_code == 204
    assert client.get("/v1/collections").json()["collections"] == []


def test_stats_endpoint(seeded):
    client, _ = seeded
    assert "collections" in client.get("/v1/stats").json()


def test_request_id_and_timing_headers(client):
    resp = client.get("/health")
    assert resp.headers["x-request-id"]
    assert float(resp.headers["x-response-time-ms"]) >= 0


def test_bearer_auth_is_enforced_when_configured(tmp_path):
    app = create_app(Database(tmp_path / "auth"), api_key="secret")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # health is public
        assert c.get("/v1/collections").status_code == 401
        assert c.get(
            "/v1/collections", headers={"authorization": "Bearer secret"}
        ).status_code == 200
        assert c.get(
            "/v1/collections", headers={"authorization": "Bearer wrong"}
        ).status_code == 401
