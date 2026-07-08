from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SOURCE = Path("data/raw/multihoprag/corpus.json")
DEFAULT_OUTPUT = Path("data/indexes/multihoprag_bge_base/chunks.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a one-chunk-per-document MultiHopRAG corpus JSONL."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def chunk_id_for(record: dict) -> str:
    key = "\n".join(
        [
            str(record.get("title", "")),
            str(record.get("source", "")),
            str(record.get("published_at", "")),
            str(record.get("url", "")),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"multihoprag:{digest}"


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(source_path.read_text(encoding="utf-8"))
    chunks = []
    seen = set()
    for record in records:
        body = str(record.get("body", "")).strip()
        if not body:
            continue
        chunk_id = chunk_id_for(record)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        chunks.append(
            {
                "doc_id": chunk_id,
                "chunk_id": chunk_id,
                "title": record.get("title"),
                "source": record.get("source"),
                "author": record.get("author"),
                "category": record.get("category"),
                "published_at": record.get("published_at"),
                "url": record.get("url"),
                "text": body,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
