"""Run a real three-node Annex cluster over HTTP, in three processes.

Each node serves the ordinary Annex API *plus* ``/v1/raft/message``, and the
nodes talk to each other with :class:`~annex.cluster.HttpTransport`.

    # terminal 1
    python examples/http_cluster.py --id n1 --port 9101
    # terminal 2
    python examples/http_cluster.py --id n2 --port 9102
    # terminal 3
    python examples/http_cluster.py --id n3 --port 9103

Then watch an election and replicate a write::

    curl -s localhost:9101/v1/raft/status | python -m json.tool
    curl -s -X POST localhost:9101/v1/collections/kb/documents \
         -H 'content-type: application/json' \
         -d '{"documents":[{"id":"a","vector":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]}]}'

Writes sent to a follower are refused with the leader's id, exactly as a Raft
cluster should behave.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from annex import Database
from annex.cluster import CollectionStateMachine, HttpTransport, RaftNode, RaftService
from annex.server.app import create_app

DEFAULT_CLUSTER = {"n1": "http://127.0.0.1:9101", "n2": "http://127.0.0.1:9102",
                   "n3": "http://127.0.0.1:9103"}
DIM = 8


def build(node_id: str, port: int, cluster: dict[str, str], data_dir: str):
    db = Database(f"{data_dir}/{node_id}")
    collection = db.create_collection("kb", dim=DIM)
    machine = CollectionStateMachine(collection)
    node = RaftNode(
        node_id,
        sorted(cluster),
        apply_fn=machine,
        snapshot_fn=machine.snapshot,
        restore_fn=machine.restore,
        election_timeout=(600, 1200),   # generous: real sockets, real latency
        heartbeat_interval=200,
    )
    peers = {nid: url for nid, url in cluster.items() if nid != node_id}
    service = RaftService(node, HttpTransport(peers, timeout=0.4))
    app = create_app(db, raft=service)

    @app.on_event("startup")
    def _start() -> None:
        service.start(interval_ms=100)
        logging.getLogger("annex.cluster").info("%s: raft driver started", node_id)

    @app.on_event("shutdown")
    def _stop() -> None:
        service.stop()

    return app, port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Annex cluster member")
    parser.add_argument("--id", required=True, choices=sorted(DEFAULT_CLUSTER))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data", default="./cluster-data")
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(message)s")
    app, port = build(args.id, args.port, DEFAULT_CLUSTER, args.data)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
