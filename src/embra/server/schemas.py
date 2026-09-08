"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    dim: int = Field(..., ge=1, le=16384)
    metric: str = "cosine"
    index: str = "hnsw"
    enable_text_index: bool = True

    @field_validator("metric")
    @classmethod
    def _metric(cls, v: str) -> str:
        if v not in ("cosine", "l2", "dot"):
            raise ValueError("metric must be one of cosine, l2, dot")
        return v

    @field_validator("index")
    @classmethod
    def _index(cls, v: str) -> str:
        if v not in ("flat", "hnsw", "hnsw_pq"):
            raise ValueError("index must be one of flat, hnsw, hnsw_pq")
        return v


class DocumentIn(BaseModel):
    id: str = Field(..., min_length=1)
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None


class UpsertRequest(BaseModel):
    documents: list[DocumentIn] = Field(..., min_length=1, max_length=10_000)


class UpsertResponse(BaseModel):
    upserted: int
    seq: int


class SearchRequest(BaseModel):
    vector: list[float] | None = None
    text: str | None = None
    k: int = Field(10, ge=1, le=1000)
    filter: dict[str, Any] | None = None
    ef: int | None = Field(None, ge=1, le=4096)
    weights: list[float] = Field(default_factory=lambda: [1.0, 1.0])
    include_vectors: bool = False
    explain: bool = True

    @field_validator("text")
    @classmethod
    def _at_least_one(cls, v, info):
        return v


class HitOut(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    text: str | None = None
    vector: list[float] | None = None


class SearchResponse(BaseModel):
    hits: list[HitOut]
    took_ms: float
    candidates_scanned: int
    plan: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str
