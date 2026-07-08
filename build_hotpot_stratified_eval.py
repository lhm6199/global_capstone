from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path("data/raw/hotpot/hotpot_dev_fullwiki_v1.json")
DEFAULT_OUTPUT = Path("data/eval/hotpot_stratified_30.json")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a stratified 30-question HotpotQA eval set."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


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


def stable_random(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def answer_category(answer: str) -> str:
    if answer.strip().lower() in {"yes", "no"}:
        return "yesno"
    return "span"


def build_candidate(item: dict, source_path: Path) -> dict | None:
    answer = str(item.get("answer", "")).strip()
    supporting_titles = sorted({title for title, _ in item.get("supporting_facts", [])})
    if not answer:
        return None
    if len(supporting_titles) < 2:
        return None

    context_by_title = {
        title: " ".join(sentence.strip() for sentence in sentences if sentence.strip()).strip()
        for title, sentences in item.get("context", [])
    }
    if any(not context_by_title.get(title) for title in supporting_titles):
        return None

    return {
        "id": item["_id"],
        "dataset": "hotpotqa",
        "question": item["question"],
        "answers": [answer],
        "answer": answer,
        "type": item["type"],
        "level": item["level"],
        "supporting_titles": supporting_titles,
        "supporting_facts": item["supporting_facts"],
        "source_split": "dev" if "dev" in source_path.name else "train",
        "answer_category": answer_category(answer),
        "_supporting_title_count": len(supporting_titles),
        "_supporting_fact_count": len(item.get("supporting_facts", [])),
    }


def rank_candidates(candidates: list[dict], seed: int) -> list[dict]:
    return sorted(
        candidates,
        key=lambda item: (
            item["answer_category"] == "yesno",
            abs(item["_supporting_title_count"] - 2),
            abs(item["_supporting_fact_count"] - 2),
            stable_random(item["id"], seed),
        ),
    )


def main(argv=None):
    global args
    args = parse_args(argv)
    source_path = Path(args.source)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(source_path.read_text(encoding="utf-8"))
    candidates = []
    for item in records:
        candidate = build_candidate(item, source_path)
        if candidate is not None:
            candidates.append(candidate)

    if len(candidates) < args.sample_size:
        raise ValueError(
            f"Not enough eligible HotpotQA candidates: need {args.sample_size}, found {len(candidates)}"
        )

    level_counter = Counter(item["level"] for item in candidates)
    type_counter = Counter(item["type"] for item in candidates)
    level_targets = proportional_allocation(
        sorted(level_counter),
        args.sample_size,
        capacities=level_counter,
        weights=level_counter,
    )

    by_level_type = defaultdict(list)
    for item in candidates:
        by_level_type[(item["level"], item["type"])].append(item)

    type_targets = {}
    for level, level_target in level_targets.items():
        keys = sorted(type_counter)
        capacities = {
            key: len(by_level_type[(level, key)])
            for key in keys
        }
        weights = capacities
        allocated = proportional_allocation(keys, level_target, capacities, weights)
        for key, value in allocated.items():
            type_targets[(level, key)] = value

    selected = []
    rng = random.Random(args.seed)
    for key, target in sorted(type_targets.items()):
        bucket = rank_candidates(by_level_type[key], args.seed)
        yesno_items = [item for item in bucket if item["answer_category"] == "yesno"]
        span_items = [item for item in bucket if item["answer_category"] == "span"]

        observed_yesno_ratio = len(yesno_items) / len(bucket) if bucket else 0.0
        yesno_cap = min(len(yesno_items), int(math.ceil(target * min(observed_yesno_ratio, 0.15))))
        chosen = span_items[: target - yesno_cap] + yesno_items[:yesno_cap]
        if len(chosen) < target:
            leftovers = [
                item for item in bucket
                if item["id"] not in {picked["id"] for picked in chosen}
            ]
            chosen.extend(leftovers[: target - len(chosen)])
        rng.shuffle(chosen)
        selected.extend(chosen[:target])

    if len(selected) < args.sample_size:
        selected_ids = {item["id"] for item in selected}
        leftovers = [
            item for item in rank_candidates(candidates, args.seed)
            if item["id"] not in selected_ids
        ]
        selected.extend(leftovers[: args.sample_size - len(selected)])

    selected = rank_candidates(selected, args.seed)[: args.sample_size]
    payload = []
    for item in selected:
        payload.append(
            {
                "id": item["id"],
                "dataset": item["dataset"],
                "question": item["question"],
                "answers": item["answers"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_titles": item["supporting_titles"],
                "supporting_facts": item["supporting_facts"],
                "source_split": item["source_split"],
            }
        )

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(payload)} items to {output_path}")
    print("Level distribution:", dict(Counter(item["level"] for item in payload)))
    print("Type distribution:", dict(Counter(item["type"] for item in payload)))
    print(
        "Answer categories:",
        dict(Counter(answer_category(item["answer"]) for item in payload)),
    )


if __name__ == "__main__":
    main()
