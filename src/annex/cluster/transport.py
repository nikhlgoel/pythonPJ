"""Transports that move Raft envelopes between nodes.

Two implementations share one interface:

``InProcessTransport``
    Hands dataclasses directly to a peer's :class:`RaftService`.  Used for tests
    and for a single-process multi-replica deployment.

``HttpTransport``
    Encodes envelopes with :mod:`annex.cluster.wire` and POSTs them to a peer's
    ``/v1/raft/message`` endpoint.  Replies come back in the HTTP response body
    and are fed straight back into the local node, so one round trip carries a
    request and its reply.

A transport is deliberately allowed to *drop* messages: Raft is designed for an
unreliable network, so a failed POST is a dropped packet, never an exception
that propagates into the state machine.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Protocol

from .raft import Message, NodeId
from .wire import decode_message, encode_message

log = logging.getLogger("annex.cluster.transport")


class Receiver(Protocol):
    """Anything that can accept an envelope and return replies."""

    def receive(self, message: Message) -> list[Message]: ...


class Transport(ABC):
    """Delivers envelopes and returns any replies to feed back locally."""

    @abstractmethod
    def send(self, messages: Iterable[Message]) -> list[Message]:
        """Deliver ``messages``; return replies for the local node to handle."""

    def close(self) -> None:  # pragma: no cover - default no-op
        return


class InProcessTransport(Transport):
    """Direct hand-off between services living in one process."""

    def __init__(self) -> None:
        self.registry: dict[NodeId, Receiver] = {}
        self.dropped: set[NodeId] = set()
        self.delivered = 0

    def register(self, node_id: NodeId, service: Receiver) -> None:
        self.registry[node_id] = service

    def isolate(self, *node_ids: NodeId) -> None:
        """Simulate a partition: traffic to these nodes is silently dropped."""
        self.dropped.update(node_ids)

    def heal(self) -> None:
        self.dropped.clear()

    def send(self, messages: Iterable[Message]) -> list[Message]:
        replies: list[Message] = []
        for msg in messages:
            target = self.registry.get(msg.dst)
            if target is None or msg.dst in self.dropped or msg.src in self.dropped:
                continue
            self.delivered += 1
            replies.extend(target.receive(msg))
        return replies


class HttpTransport(Transport):
    """POSTs envelopes to peers over HTTP.

    ``peers`` maps node ids to base URLs.  ``client_factory`` exists so the tests
    can inject a FastAPI ``TestClient`` instead of opening real sockets - the
    codec and the endpoint are exercised either way.
    """

    def __init__(
        self,
        peers: dict[NodeId, str],
        *,
        client=None,
        timeout: float = 0.5,
        path: str = "/v1/raft/message",
    ) -> None:
        self.peers = dict(peers)
        self.timeout = timeout
        self.path = path
        self.dropped = 0
        if client is None:  # pragma: no cover - exercised in real deployments
            import httpx

            client = httpx.Client(timeout=timeout)
        self.client = client

    def send(self, messages: Iterable[Message]) -> list[Message]:
        replies: list[Message] = []
        for msg in messages:
            base = self.peers.get(msg.dst)
            if base is None:
                continue
            try:
                response = self.client.post(
                    f"{base.rstrip('/')}{self.path}", json=encode_message(msg)
                )
                if response.status_code >= 400:
                    self.dropped += 1
                    continue
                for raw in response.json().get("replies", []):
                    replies.append(decode_message(raw))
            except Exception as exc:  # noqa: BLE001 - an unreachable peer is a dropped packet
                self.dropped += 1
                log.debug("raft: dropping message to %s: %s", msg.dst, exc)
        return replies

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if closer is not None:  # pragma: no branch
            closer()
