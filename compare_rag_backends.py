from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path


DEFAULT_PROMPT_TEMPLATE = """Answer the question using only the provided context.
If the context is insufficient, say so briefly.

Context:
{context}

Question: {question}
Answer:"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare FAISS and MiniRAG backends on SQuAD.")
    parser.add_argument("--dataset", default="squad", choices=["squad"])
    parser.add_argument("--eval-file", default="data/eval/squad_diverse_30.json")
    parser.add_argument("--rag-index-dir", default="data/indexes/squad_bge_base")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["faiss_naive", "minirag_naive", "minirag_light", "minirag_mini"],
    )
    parser.add_argument("--output", default="outputs/squad_backend_compare.json")
    parser.add_argument("--summary-output", default="outputs/squad_backend_compare_summary.md")
    parser.add_argument("--minirag-working-dir", default="data/indexes/minirag_squad_bge_base")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-embedding-batch-size", type=int, default=32)
    parser.add_argument("--rag-device", default=None)
    parser.add_argument("--rebuild-minirag-index", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--load_quant", required=True)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--awq_backend", choices=["auto", "kernel", "torch_fallback"], default="auto")
    parser.add_argument("--cache_dequantized_weights", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args(argv)


def load_eval_set(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"Eval file must contain a non-empty JSON list: {path}")
    return items


def render_prompt(question: str, context: str) -> str:
    return DEFAULT_PROMPT_TEMPLATE.format(question=question.strip(), context=context.strip())


class FaissNaiveBackend:
    def __init__(self, index_dir, embedding_model, batch_size, local_files_only, device, top_k):
        from rag.prompting import RagPromptBuilder
        from rag.retrieval import RagRetriever

        self.retriever = RagRetriever.from_index_dir(
            index_dir,
            embedding_model,
            batch_size=batch_size,
            local_files_only=local_files_only,
            device=device,
        )
        self.prompt_builder = RagPromptBuilder.from_file(None)
        self.top_k = top_k

    def prepare(self, question: str) -> dict:
        started = time.perf_counter()
        search = self.retriever.search_with_stats(question, self.top_k)
        retrieved = search["results"]
        context = self.prompt_builder.render_documents(retrieved)
        metrics = dict(search["metrics"])
        metrics["rag_prepare_ms"] = (time.perf_counter() - started) * 1000.0
        return {"context": context, "retrieved": retrieved, "metrics": metrics}


def build_minirag_llm(model, tokenizer, device, generation_config):
    from awq_runtime import generate_from_prompt

    async def llm_complete(prompt, system_prompt=None, history_messages=None, **_kwargs):
        segments = []
        if system_prompt:
            segments.append(system_prompt.strip())
        if history_messages:
            segments.extend(
                f"{message['role']}: {message['content']}" for message in history_messages
            )
        segments.append(prompt.strip())
        full_prompt = "\n\n".join(segment for segment in segments if segment)
        result = await asyncio.to_thread(
            generate_from_prompt, model, tokenizer, device, full_prompt, generation_config
        )
        return result["text"]

    return llm_complete


def build_backends(args, model, tokenizer, device, generation_config):
    from rag.minirag_adapter import MiniRAGAdapter
    from rag.retrieval import load_chunks_jsonl

    chunk_records = load_chunks_jsonl(Path(args.rag_index_dir) / "chunks.jsonl")
    minirag_llm = build_minirag_llm(model, tokenizer, device, generation_config)
    backends = {}
    if "faiss_naive" in args.backends:
        backends["faiss_naive"] = FaissNaiveBackend(
            args.rag_index_dir,
            args.embedding_model,
            args.rag_embedding_batch_size,
            args.local_files_only,
            args.rag_device,
            args.rag_top_k,
        )
    for backend_name, mode in (
        ("minirag_naive", "naive"),
        ("minirag_light", "light"),
        ("minirag_mini", "mini"),
    ):
        if backend_name not in args.backends:
            continue
        backends[backend_name] = MiniRAGAdapter(
            working_dir=Path(args.minirag_working_dir) / mode,
            embedding_model=args.embedding_model,
            llm_complete=minirag_llm,
            chunk_records=chunk_records,
            top_k=args.rag_top_k,
            local_files_only=args.local_files_only,
            embedding_batch_size=args.rag_embedding_batch_size,
            rebuild_index=args.rebuild_minirag_index,
            embedding_device=args.rag_device,
        )
    return backends


def build_retrieval_metrics(eval_item: dict, retrieved: list[dict], top_k: int) -> dict:
    source_chunk_id = eval_item.get("source_chunk_id")
    chunk_ids = [item.get("chunk_id") for item in retrieved[:top_k]]
    return {
        "top1_hit": bool(chunk_ids[:1] and chunk_ids[0] == source_chunk_id),
        "top3_hit": source_chunk_id in chunk_ids[:3],
        "retrieved_top_k": len(chunk_ids),
    }


def run_comparison(args):
    from awq_runtime import generate_from_prompt, load_awq_runtime, make_runtime_config
    from rag.eval_utils import build_correctness_metrics

    eval_items = load_eval_set(Path(args.eval_file))
    runtime_config = make_runtime_config(
        model_path=args.model_path,
        load_quant=args.load_quant,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        device=args.device,
        awq_backend=args.awq_backend,
        cache_dequantized_weights=args.cache_dequantized_weights,
        local_files_only=args.local_files_only,
    )
    model, tokenizer, device, awq_backend, load_timings = load_awq_runtime(runtime_config)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    backends = build_backends(args, model, tokenizer, device, generation_config)

    rows = []
    for backend_name, backend in backends.items():
        for eval_item in eval_items:
            prepare_started = time.perf_counter()
            prepared = backend.prepare(eval_item["question"])
            prompt = render_prompt(eval_item["question"], prepared["context"])
            generation = generate_from_prompt(
                model, tokenizer, device, prompt, generation_config
            )
            total_prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
            generation_metrics = {
                "retrieved_tokens": len(
                    tokenizer(prepared["context"], add_special_tokens=False)["input_ids"]
                ),
                "prompt_tokens": generation["prompt_tokens"],
                "answer_tokens": generation["answer_tokens"],
                "generate_ms": generation["timings"]["generate"] * 1000.0,
                "tokens_per_second": generation["timings"]["tokens_per_second"],
                "rag_prepare_ms": prepared["metrics"].get("rag_prepare_ms", total_prepare_ms),
            }
            for metric_name in ("embed_query_ms", "faiss_search_ms", "minirag_retrieval_ms"):
                if metric_name in prepared["metrics"]:
                    generation_metrics[metric_name] = prepared["metrics"][metric_name]
            generation_metrics["retrieval_latency_ms"] = (
                generation_metrics.get("embed_query_ms", 0.0)
                + generation_metrics.get("faiss_search_ms", 0.0)
                + generation_metrics.get("minirag_retrieval_ms", 0.0)
            )
            row = {
                "dataset": args.dataset,
                "backend": backend_name,
                "query_id": eval_item["id"],
                "question": eval_item["question"],
                "gold_answers": eval_item["answers"],
                "retrieved": prepared["retrieved"],
                "answer": generation["text"],
                "retrieval_metrics": build_retrieval_metrics(
                    eval_item, prepared["retrieved"], args.rag_top_k
                ),
                "generation_metrics": generation_metrics,
                "correctness_metrics": build_correctness_metrics(
                    generation["text"], eval_item["answers"]
                ),
            }
            rows.append(row)

    metadata = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "load_quant": args.load_quant,
        "awq_backend": awq_backend,
        "embedding_model": args.embedding_model,
        "prompt_template": "inline:DEFAULT_PROMPT_TEMPLATE",
        "backends": list(backends),
        "top_k": args.rag_top_k,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": generation_config,
        "load_timings": load_timings,
    }
    return {"metadata": metadata, "results": rows}


def _mean(rows, selector):
    values = [selector(row) for row in rows]
    return statistics.fmean(values) if values else 0.0


def build_summary(results: list[dict]) -> str:
    by_backend = {}
    for row in results:
        by_backend.setdefault(row["backend"], []).append(row)

    lines = ["# SQuAD Backend Comparison", "", "## Quality", "", "| backend | EM | F1 | contains_gold | top3_hit |", "| --- | ---: | ---: | ---: | ---: |"]
    for backend, rows in by_backend.items():
        lines.append(
            "| {backend} | {em:.3f} | {f1:.3f} | {contains:.3f} | {hit:.3f} |".format(
                backend=backend,
                em=_mean(rows, lambda row: row["correctness_metrics"]["normalized_exact_match"]),
                f1=_mean(rows, lambda row: row["correctness_metrics"]["token_f1"]),
                contains=_mean(rows, lambda row: float(row["correctness_metrics"]["contains_gold"])),
                hit=_mean(rows, lambda row: float(row["retrieval_metrics"]["top3_hit"])),
            )
        )

    lines.extend(
        [
            "",
            "## System",
            "",
            "| backend | retrieval_ms | generation_ms | total_ms | prompt_tokens |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend, rows in by_backend.items():
        lines.append(
            "| {backend} | {retrieval:.1f} | {generation:.1f} | {total:.1f} | {prompt:.1f} |".format(
                backend=backend,
                retrieval=_mean(rows, lambda row: row["generation_metrics"]["retrieval_latency_ms"]),
                generation=_mean(rows, lambda row: row["generation_metrics"]["generate_ms"]),
                total=_mean(
                    rows,
                    lambda row: row["generation_metrics"]["rag_prepare_ms"]
                    + row["generation_metrics"]["generate_ms"],
                ),
                prompt=_mean(rows, lambda row: row["generation_metrics"]["prompt_tokens"]),
            )
        )

    lines.extend(["", "## Examples", ""])
    for backend, rows in by_backend.items():
        successes = [
            row
            for row in rows
            if row["correctness_metrics"]["contains_gold"]
            and row["retrieval_metrics"]["top3_hit"]
        ][:3]
        failures = [
            row
            for row in rows
            if not row["correctness_metrics"]["contains_gold"]
        ][:3]
        lines.append(f"### {backend}")
        lines.append("")
        lines.append("Success cases:")
        for row in successes:
            lines.append(f"- {row['query_id']}: {row['question']} -> {row['answer']}")
        lines.append("Failure cases:")
        for row in failures:
            lines.append(f"- {row['query_id']}: {row['question']} -> {row['answer']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv=None):
    args = parse_args(argv)
    result = run_comparison(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(result["results"])
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Wrote results to {output_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
