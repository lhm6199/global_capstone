from __future__ import annotations


def count_text_tokens(tokenizer, text: str) -> int:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    return len(token_ids)


def build_rag_metrics(
    tokenizer,
    turn: dict,
    *,
    answer_tokens: int | None = None,
    rag_prepare_s: float | None = None,
) -> dict:
    metrics = dict(turn.get("metrics", {}))
    metrics["retrieved_tokens"] = count_text_tokens(tokenizer, turn["documents"])
    metrics["prompt_tokens"] = count_text_tokens(tokenizer, turn["prompt"])
    if answer_tokens is not None:
        metrics["answer_tokens"] = int(answer_tokens)
    if rag_prepare_s is not None:
        metrics["rag_prepare_ms"] = rag_prepare_s * 1000.0
    return metrics


def print_metrics(metrics: dict) -> None:
    if not metrics:
        return
    print("\n[metrics]")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.2f}")
        else:
            print(f"{name}: {value}")
