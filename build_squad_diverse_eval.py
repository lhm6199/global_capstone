from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("/home/jinse/projects/GlobalCapstone/naiveRAG/data/raw/squad/train-v1.1.json")
DEFAULT_CHUNKS = Path("data/indexes/squad_bge_base/chunks.jsonl")
DEFAULT_OUTPUT = Path("data/eval/squad_diverse_30.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build a diverse 30-question SQuAD eval set.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--chunks-jsonl", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=30)
    return parser.parse_args(argv)


def load_chunk_mapping(chunks_path: Path) -> dict[str, str]:
    qa_to_chunk = {}
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            chunk = json.loads(line)
            for qa_id in chunk.get("qa_ids", []):
                qa_to_chunk[qa_id] = chunk["chunk_id"]
    return qa_to_chunk


def answerable_score(item: dict) -> tuple:
    question = item["question"].lower()
    first_answer = item["answers"][0]
    return (
        len(item["answers"]) != 1,
        len(first_answer.split()) > 7,
        question.startswith("why "),
        question.startswith("how "),
        " or " in question,
        "according to" in question,
        len(question),
    )


def build_eval_items(source_path: Path, chunks_path: Path, limit: int) -> list[dict]:
    qa_to_chunk = load_chunk_mapping(chunks_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))

    by_title = defaultdict(list)
    for article in source["data"]:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                if qa["id"] not in qa_to_chunk:
                    continue
                answers = []
                for answer in qa.get("answers", []):
                    text = answer.get("text", "").strip()
                    if text and text not in answers:
                        answers.append(text)
                if not answers:
                    continue
                by_title[title].append(
                    {
                        "id": qa["id"],
                        "title": title,
                        "question": qa["question"].strip(),
                        "answers": answers,
                        "context": context,
                        "source_chunk_id": qa_to_chunk[qa["id"]],
                    }
                )

    titles = sorted(by_title)
    stride = max(1, len(titles) // limit)
    selected = []
    for index in range(0, len(titles), stride):
        title = titles[index]
        selected.append(sorted(by_title[title], key=answerable_score)[0])
        if len(selected) >= limit:
            break
    return selected


def main(argv=None):
    args = parse_args(argv)
    items = build_eval_items(Path(args.source), Path(args.chunks_jsonl), args.limit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {output}")


if __name__ == "__main__":
    main()
