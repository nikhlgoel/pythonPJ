# Storage format and durability

## Directory layout

```
<database>/
└── <collection>/
    ├── collection.json          # immutable CollectionConfig
    └── data/
        ├── MANIFEST.json        # segment list + last sequence number
        ├── wal.log              # append-only write-ahead log
        └── seg-000000-*.aseg    # immutable segments
```

## WAL frame

```
offset  size  field
0       4     magic  = "AWAL"
4       2     version (u16)
6       1     type    (1=PUT 2=DELETE 3=CHECKPOINT 4=TXN_COMMIT)
7       1     flags
8       8     seq     (u64, the engine's logical clock)
16      4     payload length (u32)
20      4     crc32 over bytes [0,20) ++ payload
24      n     payload
```

`PUT` payload: `len+id | dim | float32[dim] | len+metadata-json | len+text | has_text`.
`DELETE` payload: `len+id`.

Covering the header in the CRC means a bit-flip in `seq` or `length` is caught,
not just a corrupted payload.

## Segment file

```
"ASEG" | u32 header-length | header-json | float32[n,dim] | jsonl entries
```

The header carries `dim`, record and vector counts, `min_seq`, `max_seq` and a
CRC32 of the vector block. Entries are one JSON object per line
(`id, seq, deleted, metadata, text, row`), where `row` indexes the vector block
(`-1` for a tombstone). Segments are written to `*.tmp`, `fsync`-ed, then
`os.replace`-d — atomic on POSIX.

## Durability guarantees

| Guarantee | Mechanism |
|---|---|
| An acknowledged write survives a process crash | WAL append + `flush()` before returning |
| An acknowledged write survives a machine crash | `StorageConfig(wal_sync=True)` → `fsync` per append |
| A torn write never corrupts the database | CRC32 per frame; replay truncates at the first bad frame |
| A crash during flush never loses data | The WAL is only rotated *after* the segment is `fsync`-ed and renamed |
| Segment bit-rot is detected | Vector-block CRC verified at open |

Default is `wal_sync=False`: durable against process crash, not against power
loss. That is the correct default for an embedded engine and the wrong one for a
system of record — flip it in `StorageConfig`.

## Compaction

Flush writes one segment per memtable. When the segment count reaches
`compaction_trigger`, a **full** compaction merges every segment: for each id the
highest `seq` wins, and tombstones are dropped entirely. Dropping a tombstone is
only safe in a full compaction — a partial merge could resurrect an older row
hidden by that tombstone.

## Recovery procedure

1. Read `MANIFEST.json` (segment list, last checkpointed `seq`).
2. Replay segments oldest-first into the merged view; higher `seq` wins.
3. Replay the WAL tail; every frame re-applies its mutation and advances `seq`.
4. On a CRC failure: truncate the WAL at that offset and stop — those bytes were
   never acknowledged to a client.
5. The collection rebuilds its ANN index, BM25 index and MVCC tables from the
   recovered records, deterministically.
