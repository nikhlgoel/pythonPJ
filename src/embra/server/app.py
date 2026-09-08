"""FastAPI application factory.

Endpoints are versioned under ``/v1``.  Everything the engine can do locally is
reachable over HTTP, including ``EXPLAIN``-style query plans, snapshots stats
and maintenance operations.

Authentication is an optional bearer token (``EMBRA_API_KEY``); when the
variable is unset the server runs open, which is the right default for a local
embedded deployment and the wrong one for anything else - the startup log says
so explicitly.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..db import Database
from ..errors import ConflictError, EmbraError, NotFoundError, QueryError, SchemaError
from .schemas import (
    CreateCollectionRequest,
    HitOut,
    SearchRequest,
    SearchResponse,
    UpsertRequest,
    UpsertResponse,
)

log = logging.getLogger("embra.server")

_STATUS = {
    NotFoundError: 404,
    SchemaError: 400,
    QueryError: 400,
    ConflictError: 409,
}


def create_app(db: Database, *, api_key: str | None = None) -> FastAPI:
    """Build an ASGI app serving ``db``."""
    app = FastAPI(
        title="Embra",
        version=__version__,
        description="Embedded, transactional, vector-native search engine.",
    )
    app.state.db = db
    app.state.api_key = api_key or os.getenv("EMBRA_API_KEY")
    if not app.state.api_key:
        log.warning("EMBRA_API_KEY is unset - the API is unauthenticated")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        expected = app.state.api_key
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    auth = [Depends(require_auth)]

    # -- middleware ----------------------------------------------------
    @app.middleware("http")
    async def observability(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        response.headers["x-response-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    @app.exception_handler(EmbraError)
    async def engine_error(_request: Request, exc: EmbraError):
        status = next((s for t, s in _STATUS.items() if isinstance(exc, t)), 500)
        return JSONResponse(
            status_code=status,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    def collection(name: str):
        try:
            return db.open_collection(name)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # -- routes --------------------------------------------------------
    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "collections": db.list_collections()}

    @app.get("/v1/stats", tags=["ops"], dependencies=auth)
    def stats() -> dict[str, Any]:
        return db.stats()

    @app.get("/v1/collections", tags=["collections"], dependencies=auth)
    def list_collections() -> dict[str, Any]:
        return {"collections": db.list_collections()}

    @app.post("/v1/collections", status_code=201, tags=["collections"], dependencies=auth)
    def create_collection(body: CreateCollectionRequest) -> dict[str, Any]:
        coll = db.create_collection(
            body.name,
            body.dim,
            metric=body.metric,
            index=body.index,
            enable_text_index=body.enable_text_index,
        )
        return coll.stats()

    @app.get("/v1/collections/{name}", tags=["collections"], dependencies=auth)
    def get_collection(name: str) -> dict[str, Any]:
        return collection(name).stats()

    @app.delete("/v1/collections/{name}", status_code=204, tags=["collections"], dependencies=auth)
    def drop_collection(name: str) -> None:
        db.drop_collection(name)

    @app.post(
        "/v1/collections/{name}/documents",
        response_model=UpsertResponse,
        tags=["documents"],
        dependencies=auth,
    )
    def upsert(name: str, body: UpsertRequest) -> UpsertResponse:
        coll = collection(name)
        count = coll.upsert_many(
            {"id": d.id, "vector": d.vector, "metadata": d.metadata, "text": d.text}
            for d in body.documents
        )
        return UpsertResponse(upserted=count, seq=coll.current_seq)

    @app.get("/v1/collections/{name}/documents/{doc_id}", tags=["documents"], dependencies=auth)
    def get_document(name: str, doc_id: str, include_vector: bool = False) -> dict[str, Any]:
        hit = collection(name).get(doc_id, include_vector=include_vector)
        if hit is None:
            raise HTTPException(status_code=404, detail=f"no such document: {doc_id}")
        payload = hit.as_dict()
        if include_vector and hit.vector is not None:
            payload["vector"] = hit.vector.tolist()
        return payload

    @app.delete(
        "/v1/collections/{name}/documents/{doc_id}",
        status_code=204,
        tags=["documents"],
        dependencies=auth,
    )
    def delete_document(name: str, doc_id: str) -> None:
        if not collection(name).delete(doc_id):
            raise HTTPException(status_code=404, detail=f"no such document: {doc_id}")

    @app.post(
        "/v1/collections/{name}/search",
        response_model=SearchResponse,
        tags=["search"],
        dependencies=auth,
    )
    def search(name: str, body: SearchRequest) -> SearchResponse:
        if body.vector is None and not body.text:
            raise HTTPException(status_code=400, detail="supply 'vector', 'text' or both")
        result = collection(name).search(
            body.vector,
            text=body.text,
            k=body.k,
            filter=body.filter,
            ef=body.ef,
            weights=body.weights,
            include_vectors=body.include_vectors,
            explain=body.explain,
        )
        return SearchResponse(
            hits=[
                HitOut(
                    id=h.id,
                    score=h.score,
                    metadata=h.metadata,
                    text=h.text,
                    vector=None if h.vector is None else h.vector.tolist(),
                )
                for h in result.hits
            ],
            took_ms=result.took_ms,
            candidates_scanned=result.candidates_scanned,
            plan=result.plan,
        )

    @app.post("/v1/collections/{name}/vacuum", tags=["maintenance"], dependencies=auth)
    def vacuum(name: str) -> dict[str, int]:
        return {"reclaimed": collection(name).vacuum()}

    @app.post("/v1/collections/{name}/rebuild", tags=["maintenance"], dependencies=auth)
    def rebuild(name: str) -> dict[str, Any]:
        coll = collection(name)
        coll.rebuild_index()
        return coll.stats()

    @app.post("/v1/collections/{name}/flush", tags=["maintenance"], dependencies=auth)
    def flush(name: str) -> dict[str, Any]:
        coll = collection(name)
        coll.flush()
        return coll.stats()["storage"]

    return app


def create_default_app() -> FastAPI:
    """Factory for ``uvicorn embra.server.app:create_default_app --factory``."""
    logging.basicConfig(level=os.getenv("EMBRA_LOG_LEVEL", "INFO"))
    return create_app(Database(os.getenv("EMBRA_DATA_DIR", ".embra-data")))
