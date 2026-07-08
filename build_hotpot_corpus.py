from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SOURCE = Path("data/raw/hotpot/hotpot_dev_fullwiki_v1.json")
DEFAULT_OUTPUT = Path("data/indexes/hotpotqa_dev_bge_base/chunks.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a deduplicated paragraph-level HotpotQA corpus JSONL."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def chunk_id_for(title: str, text: str) -> str:
    digest = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]
    return f"hotpot:{digest}"


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(source_path.read_text(encoding="utf-8"))
    seen = set()
    chunks = []
    for item in records:
        for title, sentences in item.get("context", []):
            text = " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
            if not text:
                continue
            key = (title, text)
            if key in seen:
                continue
            seen.add(key)
            chunk_id = chunk_id_for(title, text)
            chunks.append(
                {
                    "doc_id": chunk_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "source": str(source_path),
                    "text": text,
                }
            )

    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Wrote {len(chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
