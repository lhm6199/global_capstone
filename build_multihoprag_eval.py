from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SOURCE = Path("data/raw/multihoprag/MultiHopRAG.json")
DEFAULT_OUTPUT = Path("data/eval/multihoprag_eval_full.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert the official MultiHopRAG queries into the local eval schema."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(source_path.read_text(encoding="utf-8"))
    payload = []
    for index, item in enumerate(records):
        evidence_list = item.get("evidence_list", [])
        supporting_titles = sorted(
            {
                evidence.get("title")
                for evidence in evidence_list
                if evidence.get("title")
            }
        )
        payload.append(
            {
                "id": f"multihoprag-{index:04d}",
                "dataset": "multihoprag",
                "question": item["query"],
                "answers": [item["answer"]],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence_list": evidence_list,
                "supporting_titles": supporting_titles,
                "evidence_count": len(evidence_list),
            }
        )

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload)} items to {output_path}")


if __name__ == "__main__":
    main()
