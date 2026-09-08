"""``embra`` command line interface.

Examples
--------
::

    embra create papers --dim 384 --index hnsw
    embra ingest papers documents.jsonl --text-field abstract
    embra query papers --text "graph neural networks" --k 5
    embra stats papers
    embra bench --n 20000 --dim 128
    embra serve --port 8080
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .db import Database

app = typer.Typer(add_completion=False, help="Embra vector search engine")
console = Console()


def _db(path: str) -> Database:
    return Database(path)


DATA_OPT = typer.Option(".embra-data", "--data", "-d", help="database directory")


@app.command()
def version() -> None:
    """Print the engine version."""
    console.print(f"embra {__version__}")


@app.command("list")
def list_collections(data: str = DATA_OPT) -> None:
    """List collections."""
    names = _db(data).list_collections()
    if not names:
        console.print("[yellow]no collections[/yellow]")
        return
    for name in names:
        console.print(f"- {name}")


@app.command()
def create(
    name: str,
    dim: int = typer.Option(..., "--dim", help="vector dimensionality"),
    metric: str = typer.Option("cosine", help="cosine | l2 | dot"),
    index: str = typer.Option("hnsw", help="flat | hnsw | hnsw_pq"),
    data: str = DATA_OPT,
) -> None:
    """Create a collection."""
    coll = _db(data).create_collection(name, dim, metric=metric, index=index)
    console.print(f"[green]created[/green] {name} dim={dim} metric={metric} index={index}")
    coll.close()


@app.command()
def ingest(
    name: str,
    path: Path = typer.Argument(..., help="JSONL file with id/vector/metadata/text"),
    data: str = DATA_OPT,
    batch: int = typer.Option(1000, help="records per flush"),
) -> None:
    """Bulk-load a JSONL file into a collection."""
    coll = _db(data).open_collection(name)
    buffer: list[dict[str, Any]] = []
    total = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            buffer.append(json.loads(line))
            if len(buffer) >= batch:
                total += coll.upsert_many(buffer)
                buffer.clear()
    if buffer:
        total += coll.upsert_many(buffer)
    coll.flush()
    coll.close()
    console.print(f"[green]ingested[/green] {total} documents into {name}")


@app.command()
def query(
    name: str,
    text: str = typer.Option(None, "--text", "-t", help="lexical / hybrid query"),
    vector: str = typer.Option(None, "--vector", help="JSON array query vector"),
    k: int = typer.Option(10, "-k"),
    filter_: str = typer.Option(None, "--filter", help="JSON metadata filter"),
    ef: int = typer.Option(None, "--ef"),
    data: str = DATA_OPT,
) -> None:
    """Search a collection and print the result table plus the query plan."""
    coll = _db(data).open_collection(name)
    res = coll.search(
        json.loads(vector) if vector else None,
        text=text,
        k=k,
        filter=json.loads(filter_) if filter_ else None,
        ef=ef,
    )
    table = Table(title=f"{name}: top {len(res)} ({res.took_ms:.2f} ms)")
    table.add_column("#", justify="right")
    table.add_column("id")
    table.add_column("score", justify="right")
    table.add_column("text", overflow="ellipsis", max_width=48)
    for i, hit in enumerate(res.hits, 1):
        table.add_row(str(i), hit.id, f"{hit.score:.4f}", hit.text or "")
    console.print(table)
    console.print(f"[dim]plan:[/dim] {json.dumps(res.plan)}")
    coll.close()


@app.command()
def stats(name: str = typer.Argument(None), data: str = DATA_OPT) -> None:
    """Show engine statistics."""
    db = _db(data)
    payload = db.open_collection(name).stats() if name else db.stats()
    console.print_json(json.dumps(payload, default=str))


@app.command()
def vacuum(name: str, data: str = DATA_OPT) -> None:
    """Reclaim dead MVCC versions."""
    coll = _db(data).open_collection(name)
    console.print(f"reclaimed {coll.vacuum()} versions")
    coll.close()


@app.command()
def bench(
    n: int = typer.Option(20_000, "--n"),
    dim: int = typer.Option(128, "--dim"),
    queries: int = typer.Option(200, "--queries"),
    k: int = typer.Option(10, "-k"),
) -> None:
    """Run the built-in recall / latency benchmark."""
    from .bench.suite import run_suite

    for res in run_suite(n, dim, queries, k):
        console.print(res.row())


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8080,
    data: str = DATA_OPT,
) -> None:
    """Serve the HTTP API."""
    import uvicorn

    from .server.app import create_app

    uvicorn.run(create_app(Database(data)), host=host, port=port)


def main() -> None:  # pragma: no cover - entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
