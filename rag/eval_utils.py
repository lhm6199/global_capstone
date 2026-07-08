from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_metric_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def normalized_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def contains_gold(prediction: str, ground_truths: list[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(
        answer and normalize_answer(answer) in normalized_prediction
        for answer in ground_truths
    )


def build_correctness_metrics(prediction: str, ground_truths: list[str]) -> dict:
    return {
        "contains_gold": contains_gold(prediction, ground_truths),
        "normalized_exact_match": max_metric_over_ground_truths(
            normalized_exact_match, prediction, ground_truths
        ),
        "token_f1": max_metric_over_ground_truths(token_f1_score, prediction, ground_truths),
    }
