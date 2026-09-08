"""Exception hierarchy for Embra.

Every failure mode the engine can surface is a subclass of :class:`EmbraError`,
so embedding applications can guard the whole engine with a single ``except``.
"""

from __future__ import annotations


class EmbraError(Exception):
    """Base class for every Embra failure."""


class ConfigError(EmbraError):
    """Invalid engine / collection configuration."""


class SchemaError(EmbraError):
    """A record violates the collection schema (dimension, dtype, id type)."""


class NotFoundError(EmbraError):
    """A collection, record or snapshot does not exist."""


class ConflictError(EmbraError):
    """Write-write conflict detected by the MVCC layer."""


class TransactionError(EmbraError):
    """Illegal transaction lifecycle operation (e.g. use after commit)."""


class CorruptionError(EmbraError):
    """On-disk state failed an integrity check (CRC, magic, version)."""


class IndexError_(EmbraError):
    """Index build / query failure."""


class QueryError(EmbraError):
    """Malformed query: bad filter DSL, unknown field, bad k."""


class ClusterError(EmbraError):
    """Replication / consensus failure."""
