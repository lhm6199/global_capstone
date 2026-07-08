from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("data/eval/multihoprag_eval_full.json")
DEFAULT_OUTPUT = Path("data/eval/multihoprag_test_30.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a stratified 30-question MultiHopRAG test subset."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def stable_random(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def proportional_allocation(keys, total, capacities, weights):
    if total <= 0 or not keys:
        return {key: 0 for key in keys}

    feasible = [key for key in keys if capacities.get(key, 0) > 0]
    if not feasible:
        return {key: 0 for key in keys}

    total_weight = sum(max(weights.get(key, 0.0), 0.0) for key in feasible)
    if total_weight <= 0:
        total_weight = float(len(feasible))
        weights = {key: 1.0 for key in feasible}

    raw = {
        key: total * max(weights.get(key, 0.0), 0.0) / total_weight
        for key in feasible
    }
    allocated = {
        key: min(capacities.get(key, 0), int(math.floor(raw[key])))
        for key in feasible
    }
    remaining = total - sum(allocated.values())
    order = sorted(
        feasible,
        key=lambda key: (raw[key] - math.floor(raw[key]), weights.get(key, 0.0)),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if allocated[key] >= capacities.get(key, 0):
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return {key: allocated.get(key, 0) for key in keys}


def answer_category(item: dict) -> str:
    answer = str(item["answer"]).strip().lower()
    if answer in {"yes", "no"}:
        return answer
    if answer == "insufficient information.":
        return "insufficient"
    return "span"


def rank_items(items: list[dict], seed: int) -> list[dict]:
    return sorted(
        items,
        key=lambda item: (
            item["evidence_count"],
            len(item.get("supporting_titles", [])),
            stable_random(item["id"], seed),
        ),
    )


def main(argv=None):
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(source_path.read_text(encoding="utf-8"))
    by_question_type = defaultdict(list)
    for item in records:
        by_question_type[item["question_type"]].append(item)

    question_types = sorted(by_question_type)
    capacities = {key: len(by_question_type[key]) for key in question_types}
    targets = proportional_allocation(
        question_types,
        args.sample_size,
        capacities=capacities,
        weights=capacities,
    )

    selected = []
    for question_type in question_types:
        bucket = by_question_type[question_type]
        by_answer_type = defaultdict(list)
        for item in bucket:
            by_answer_type[answer_category(item)].append(item)

        answer_keys = sorted(by_answer_type)
        answer_caps = {key: len(by_answer_type[key]) for key in answer_keys}
        answer_targets = proportional_allocation(
            answer_keys,
            targets[question_type],
            capacities=answer_caps,
            weights=answer_caps,
        )

        bucket_selected = []
        for answer_key in answer_keys:
            bucket_selected.extend(rank_items(by_answer_type[answer_key], args.seed)[: answer_targets[answer_key]])

        if len(bucket_selected) < targets[question_type]:
            picked_ids = {item["id"] for item in bucket_selected}
            leftovers = [
                item
                for item in rank_items(bucket, args.seed)
                if item["id"] not in picked_ids
            ]
            bucket_selected.extend(leftovers[: targets[question_type] - len(bucket_selected)])

        selected.extend(bucket_selected[: targets[question_type]])

    if len(selected) < args.sample_size:
        picked_ids = {item["id"] for item in selected}
        leftovers = [
            item
            for item in rank_items(records, args.seed)
            if item["id"] not in picked_ids
        ]
        selected.extend(leftovers[: args.sample_size - len(selected)])

    selected = rank_items(selected, args.seed)[: args.sample_size]
    output_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(selected)} items to {output_path}")
    print("Question types:", dict(Counter(item["question_type"] for item in selected)))
    print("Answer categories:", dict(Counter(answer_category(item) for item in selected)))


if __name__ == "__main__":
    main()
