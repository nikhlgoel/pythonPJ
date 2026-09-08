"""Exception hierarchy for Annex.

Every failure mode the engine can surface is a subclass of :class:`AnnexError`,
so embedding applications can guard the whole engine with a single ``except``.
"""

from __future__ import annotations


class AnnexError(Exception):
    """Base class for every Annex failure."""


class ConfigError(AnnexError):
    """Invalid engine / collection configuration."""


class SchemaError(AnnexError):
    """A record violates the collection schema (dimension, dtype, id type)."""


class NotFoundError(AnnexError):
    """A collection, record or snapshot does not exist."""


class ConflictError(AnnexError):
    """Write-write conflict detected by the MVCC layer."""


class TransactionError(AnnexError):
    """Illegal transaction lifecycle operation (e.g. use after commit)."""


class CorruptionError(AnnexError):
    """On-disk state failed an integrity check (CRC, magic, version)."""


class IndexError_(AnnexError):
    """Index build / query failure."""


class QueryError(AnnexError):
    """Malformed query: bad filter DSL, unknown field, bad k."""


class ClusterError(AnnexError):
    """Replication / consensus failure."""
