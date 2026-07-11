from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a SQuAD v1.1 JSON file to paragraph-level RAG chunks."
    )
    parser.add_argument("--squad-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--split-name",
        default=None,
        help="Optional split label for IDs, e.g. squad-dev. Defaults to input stem.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.squad_json)
    output_path = Path(args.output_jsonl)
    split_name = args.split_name or input_path.stem

    with input_path.open("r", encoding="utf-8") as f:
        squad = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    qa_count = 0

    with output_path.open("w", encoding="utf-8") as f:
        for article_idx, article in enumerate(squad["data"]):
            doc_id = f"{split_name}:article:{article_idx:04d}"
            title = article.get("title", "")
            for paragraph_idx, paragraph in enumerate(article.get("paragraphs", [])):
                qas = paragraph.get("qas", [])
                qa_ids = [qa["id"] for qa in qas if "id" in qa]
                row = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}:paragraph:{paragraph_idx:04d}",
                    "title": title,
                    "paragraph_id": paragraph_idx,
                    "source": str(input_path),
                    "text": paragraph["context"],
                    "qa_ids": qa_ids,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                chunk_count += 1
                qa_count += len(qa_ids)

    print(f"Wrote {chunk_count} chunks with {qa_count} QA ids to {output_path}")


if __name__ == "__main__":
    main()
