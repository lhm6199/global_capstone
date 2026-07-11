from __future__ import annotations

import argparse
import json
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from rag import RagChatTurn, RagPromptBuilder, RagRetriever

NO_RAG_PROMPT_TEMPLATE = "Question:\n{question}\n\nAnswer:"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate SQuAD RAG retrieval and optional AWQ generation."
    )
    parser.add_argument(
        "--mode",
        choices=["no_rag", "rag", "adaptive", "all"],
        default="rag",
        help="Evaluation mode. 'all' runs no_rag, fixed-k RAG, and adaptive-k RAG.",
    )
    parser.add_argument("--squad-json", default="data/raw/squad/dev-v1.1.json")
    parser.add_argument("--rag-index-dir", default="data/indexes/squad_dev_bge_base")
    parser.add_argument("--output-jsonl", default="data/eval/squad_dev_rag_eval.jsonl")
    parser.add_argument("--summary-json", default="data/eval/squad_dev_rag_eval_summary.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-prompt-template", default=None)
    parser.add_argument(
        "--no-rag-prompt-template",
        default=NO_RAG_PROMPT_TEMPLATE,
        help="Prompt template for no-RAG generation. Must contain {question}.",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rag-embedding-batch-size", type=int, default=32)
    parser.add_argument("--rag-local-files-only", action="store_true")
    parser.add_argument("--rag-device", default=None)
    parser.add_argument(
        "--adaptive-search-k",
        type=int,
        default=50,
        help="Number of FAISS candidates searched before adaptive-k filtering.",
    )
    parser.add_argument("--adaptive-ignore-head", type=float, default=0.0)
    parser.add_argument("--adaptive-ignore-tail", type=float, default=0.1)
    parser.add_argument("--adaptive-retrieve-more", type=int, default=5)
    parser.add_argument("--adaptive-min-k", type=int, default=1)
    parser.add_argument("--adaptive-max-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Measure retrieval hit rate and retrieval timing without loading a generator. Skips no_rag in --mode all.",
    )

    parser.add_argument("--model_path", default="qwen3-4b-awq-runtime")
    parser.add_argument("--load_quant", default=None)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--awq_backend",
        choices=["auto", "kernel", "torch_fallback"],
        default="auto",
    )
    parser.add_argument("--cache_dequantized_weights", action="store_true")
    return parser.parse_args(argv)


def load_squad_examples(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        squad = json.load(f)

    examples = []
    for article in squad["data"]:
        title = article.get("title", "")
        for paragraph_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                examples.append(
                    {
                        "id": qa["id"],
                        "title": title,
                        "paragraph_id": paragraph_idx,
                        "question": qa["question"],
                        "answers": [answer["text"] for answer in qa.get("answers", [])],
                        "context": context,
                    }
                )
    return examples


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_metric(prediction: str, answers: list[str], metric_fn) -> float:
    if not answers:
        return 0.0
    return max(float(metric_fn(prediction, answer)) for answer in answers)


def find_largest_gap_k(
    scores,
    *,
    ignore_head: float = 0.0,
    ignore_tail: float = 0.1,
    retrieve_more: int = 5,
    min_k: int = 1,
    max_k: int | None = None,
) -> int:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or scores.size == 0:
        return int(min_k)
    if scores.size == 1:
        return int(min_k)

    gaps = scores[:-1] - scores[1:]
    start = int(gaps.size * ignore_head)
    end = gaps.size - int(gaps.size * ignore_tail)
    start = max(0, min(start, gaps.size - 1))
    end = max(start + 1, min(end, gaps.size))
    threshold = int(np.argmax(gaps[start:end]) + start)
    base_k = threshold + 1
    k = base_k + int(retrieve_more)
    upper = scores.size if max_k is None else min(int(max_k), scores.size)
    return max(int(min_k), min(int(k), upper))


class AdaptiveRagChatTurn:
    def __init__(
        self,
        retriever,
        prompt_builder,
        *,
        search_k: int,
        ignore_head: float,
        ignore_tail: float,
        retrieve_more: int,
        min_k: int,
        max_k: int | None,
    ):
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.search_k = search_k
        self.ignore_head = ignore_head
        self.ignore_tail = ignore_tail
        self.retrieve_more = retrieve_more
        self.min_k = min_k
        self.max_k = max_k

    def prepare(self, query: str) -> dict:
        if self.search_k <= 0:
            raise ValueError("--adaptive-search-k must be positive")

        search_k = min(self.search_k, self.retriever.index.ntotal)
        embed_started = time.perf_counter()
        query_vector = np.asarray([self.retriever.embedder.embed_query(query)], dtype=np.float32)
        embed_query_ms = (time.perf_counter() - embed_started) * 1000.0

        search_started = time.perf_counter()
        scores, indices = self.retriever.index.search(query_vector, search_k)
        faiss_search_ms = (time.perf_counter() - search_started) * 1000.0

        adaptive_k = find_largest_gap_k(
            scores[0],
            ignore_head=self.ignore_head,
            ignore_tail=self.ignore_tail,
            retrieve_more=self.retrieve_more,
            min_k=self.min_k,
            max_k=self.max_k or search_k,
        )

        results = []
        for rank, (score, chunk_idx) in enumerate(
            zip(scores[0][:adaptive_k], indices[0][:adaptive_k]),
            start=1,
        ):
            if chunk_idx < 0 or chunk_idx >= len(self.retriever.chunks):
                continue
            chunk = dict(self.retriever.chunks[chunk_idx])
            chunk["rank"] = rank
            chunk["score"] = float(score)
            results.append(chunk)

        prompt, documents = self.prompt_builder.build_with_documents(query, results)
        return {
            "prompt": prompt,
            "results": results,
            "documents": documents,
            "metrics": {
                "embed_query_ms": embed_query_ms,
                "faiss_search_ms": faiss_search_ms,
                "rag_search_k": search_k,
                "adaptive_k": adaptive_k,
                "adaptive_strategy": "largest_gap",
                "adaptive_ignore_head": self.ignore_head,
                "adaptive_ignore_tail": self.ignore_tail,
                "adaptive_retrieve_more": self.retrieve_more,
            },
        }


def build_rag(args) -> RagChatTurn:
    retriever = RagRetriever.from_index_dir(
        args.rag_index_dir,
        args.embedding_model,
        batch_size=args.rag_embedding_batch_size,
        local_files_only=args.rag_local_files_only,
        device=args.rag_device,
    )
    return RagChatTurn(
        retriever=retriever,
        prompt_builder=RagPromptBuilder.from_file(args.rag_prompt_template),
        top_k=args.rag_top_k,
    )


def build_adaptive_rag(args) -> AdaptiveRagChatTurn:
    retriever = RagRetriever.from_index_dir(
        args.rag_index_dir,
        args.embedding_model,
        batch_size=args.rag_embedding_batch_size,
        local_files_only=args.rag_local_files_only,
        device=args.rag_device,
    )
    return AdaptiveRagChatTurn(
        retriever=retriever,
        prompt_builder=RagPromptBuilder.from_file(args.rag_prompt_template),
        search_k=args.adaptive_search_k,
        ignore_head=args.adaptive_ignore_head,
        ignore_tail=args.adaptive_ignore_tail,
        retrieve_more=args.adaptive_retrieve_more,
        min_k=args.adaptive_min_k,
        max_k=args.adaptive_max_k,
    )


def load_generator(args):
    if not args.load_quant:
        raise ValueError("--load_quant is required unless --retrieval-only is set.")

    import torch
    from accelerate import init_empty_weights, load_checkpoint_in_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from infer_awq import (
        build_inputs,
        cache_dequantized_weights,
        normalize_hf_model_path,
        pick_awq_backend,
        pick_device,
        replace_decoder_linears_with_awq,
        sync_device,
    )

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = args.device or pick_device()
    awq_backend = pick_awq_backend(args.awq_backend, device)
    if awq_backend == "kernel" and not str(device).startswith("cuda"):
        raise ValueError("--awq_backend kernel requires a CUDA device.")

    load_started = time.perf_counter()
    model_path = normalize_hf_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    replace_decoder_linears_with_awq(model, args.w_bit, args.q_group_size, awq_backend)
    model.tie_weights()
    sync_device(device)
    load_checkpoint_in_model(
        model,
        checkpoint=args.load_quant,
        device_map={"": device},
        offload_state_dict=True,
    )
    model = model.to(device).eval()
    if args.cache_dequantized_weights:
        if awq_backend != "torch_fallback":
            print("--cache_dequantized_weights is ignored unless --awq_backend torch_fallback.")
        else:
            cache_dequantized_weights(model)
    sync_device(device)

    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "awq_backend": awq_backend,
        "build_inputs": build_inputs,
        "sync_device": sync_device,
        "load_model_ms": (time.perf_counter() - load_started) * 1000.0,
    }


def generate_answer(generator: dict, prompt: str, args) -> dict:
    torch = generator["torch"]
    tokenizer = generator["tokenizer"]
    model = generator["model"]
    device = generator["device"]
    build_inputs = generator["build_inputs"]
    sync_device = generator["sync_device"]

    build_inputs_started = time.perf_counter()
    inputs = build_inputs(tokenizer, prompt, device)
    sync_device(device)
    build_inputs_ms = (time.perf_counter() - build_inputs_started) * 1000.0

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    generate_started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)
    sync_device(device)
    generate_ms = (time.perf_counter() - generate_started) * 1000.0

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    decode_started = time.perf_counter()
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    decode_ms = (time.perf_counter() - decode_started) * 1000.0

    return {
        "prediction": prediction,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "generated_tokens": int(generated_ids.numel()),
        "build_inputs_ms": build_inputs_ms,
        "generate_ms": generate_ms,
        "decode_ms": decode_ms,
        "generator_total_ms": build_inputs_ms + generate_ms + decode_ms,
    }


def summarize_mode(rows: list[dict], args, generator: dict | None) -> dict:
    def avg(name: str) -> float:
        values = [row[name] for row in rows if row.get(name) is not None]
        return float(np.mean(values)) if values else 0.0

    def pct(name: str) -> float:
        return avg(name) * 100.0

    summary = {
        "num_examples": len(rows),
        "mode": rows[0]["mode"] if rows else None,
        "rag_index_dir": args.rag_index_dir,
        "squad_json": args.squad_json,
        "rag_top_k": args.rag_top_k,
        "retrieval_hit_rate_percent": pct("retrieval_hit"),
        "avg_retrieval_total_ms": avg("retrieval_total_ms"),
        "avg_embed_query_ms": avg("embed_query_ms"),
        "avg_faiss_search_ms": avg("faiss_search_ms"),
        "avg_prompt_build_ms": avg("prompt_build_ms"),
        "avg_end_to_end_ms": avg("end_to_end_ms"),
        "avg_retrieved_count": avg("retrieved_count"),
        "avg_adaptive_k": avg("adaptive_k"),
    }
    if generator is not None:
        summary.update(
            {
                "model_path": args.model_path,
                "awq_backend": generator["awq_backend"],
                "load_model_ms": generator["load_model_ms"],
                "exact_match_percent": pct("exact_match"),
                "f1_percent": pct("f1"),
                "avg_build_inputs_ms": avg("build_inputs_ms"),
                "avg_generate_ms": avg("generate_ms"),
                "avg_decode_ms": avg("decode_ms"),
                "avg_generator_total_ms": avg("generator_total_ms"),
                "avg_input_tokens": avg("input_tokens"),
                "avg_generated_tokens": avg("generated_tokens"),
            }
        )
    return summary


def summarize(rows: list[dict], args, generator: dict | None) -> dict:
    modes = []
    for mode in ["no_rag", "rag", "adaptive"]:
        mode_rows = [row for row in rows if row["mode"] == mode]
        if mode_rows:
            modes.append(summarize_mode(mode_rows, args, generator))
    return {
        "num_rows": len(rows),
        "num_examples": len({row["id"] for row in rows}),
        "modes": modes,
    }


def selected_modes(args) -> list[str]:
    if args.mode == "all":
        modes = ["no_rag", "rag", "adaptive"]
    else:
        modes = [args.mode]
    if args.retrieval_only:
        modes = [mode for mode in modes if mode != "no_rag"]
    return modes


def build_mode_runners(args) -> dict:
    modes = selected_modes(args)
    runners = {}
    if "rag" in modes:
        runners["rag"] = build_rag(args)
    if "adaptive" in modes:
        runners["adaptive"] = build_adaptive_rag(args)
    return runners


def prepare_mode(mode: str, example: dict, args, runners: dict) -> tuple[str, dict | None, dict]:
    if mode == "no_rag":
        return args.no_rag_prompt_template.format(question=example["question"]), None, {}

    retrieval_started = time.perf_counter()
    turn = runners[mode].prepare(example["question"])
    retrieval_total_ms = (time.perf_counter() - retrieval_started) * 1000.0
    retrieval_metrics = turn["metrics"]
    prompt_build_ms = retrieval_total_ms
    prompt_build_ms -= float(retrieval_metrics.get("embed_query_ms", 0.0))
    prompt_build_ms -= float(retrieval_metrics.get("faiss_search_ms", 0.0))

    retrieval_hit = any(
        example["id"] in set(item.get("qa_ids", [])) for item in turn["results"]
    )
    metrics = {
        "retrieved_chunk_ids": [item.get("chunk_id") for item in turn["results"]],
        "retrieval_hit": float(retrieval_hit),
        "retrieval_total_ms": retrieval_total_ms,
        "embed_query_ms": float(retrieval_metrics.get("embed_query_ms", 0.0)),
        "faiss_search_ms": float(retrieval_metrics.get("faiss_search_ms", 0.0)),
        "prompt_build_ms": max(prompt_build_ms, 0.0),
        "retrieved_count": len(turn["results"]),
    }
    for key in (
        "rag_search_k",
        "adaptive_k",
        "adaptive_strategy",
        "adaptive_ignore_head",
        "adaptive_ignore_tail",
        "adaptive_retrieve_more",
    ):
        if key in retrieval_metrics:
            metrics[key] = retrieval_metrics[key]
    return turn["prompt"], turn, metrics


def main(argv=None):
    args = parse_args(argv)
    examples = load_squad_examples(args.squad_json)
    examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]

    modes = selected_modes(args)
    if not modes:
        raise ValueError("--retrieval-only with --mode no_rag has nothing to evaluate.")
    runners = build_mode_runners(args)
    generator = None if args.retrieval_only else load_generator(args)

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with output_path.open("w", encoding="utf-8") as f:
        for idx, example in enumerate(examples, start=1):
            for mode in modes:
                e2e_started = time.perf_counter()
                prompt, _turn, mode_metrics = prepare_mode(mode, example, args, runners)
                row = {
                    "mode": mode,
                    "id": example["id"],
                    "title": example["title"],
                    "paragraph_id": example["paragraph_id"],
                    "question": example["question"],
                    "answers": example["answers"],
                    "retrieved_chunk_ids": None,
                    "retrieval_hit": None,
                    "retrieval_total_ms": None,
                    "embed_query_ms": None,
                    "faiss_search_ms": None,
                    "prompt_build_ms": None,
                    "retrieved_count": 0,
                }
                row.update(mode_metrics)

                if generator is not None:
                    generation = generate_answer(generator, prompt, args)
                    prediction = generation["prediction"]
                    row.update(generation)
                    row["exact_match"] = max_metric(prediction, example["answers"], exact_match)
                    row["f1"] = max_metric(prediction, example["answers"], f1_score)
                else:
                    row["prediction"] = None
                    row["exact_match"] = None
                    row["f1"] = None

                row["end_to_end_ms"] = (time.perf_counter() - e2e_started) * 1000.0
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if idx == 1 or idx % 10 == 0 or idx == len(examples):
                print(f"evaluated {idx}/{len(examples)}")

    summary = summarize(rows, args, generator)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
