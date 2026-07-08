from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from awq_runtime import generate_from_prompt, load_awq_runtime, make_runtime_config
from rag.minirag_adapter import MiniRAGAdapter
from rag.retrieval import load_chunks_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a MiniRAG working directory from an existing chunks.jsonl corpus."
    )
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--rag-top-k", type=int, default=3)
    parser.add_argument("--rag-device", default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model_path", default="qwen3-4b-awq-runtime")
    parser.add_argument("--load_quant", required=True)
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--awq_backend", choices=["auto", "kernel", "torch_fallback"], default="auto")
    parser.add_argument("--cache_dequantized_weights", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args(argv)


def build_minirag_llm(model, tokenizer, device, generation_config):
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


def main(argv=None):
    args = parse_args(argv)
    chunk_records = load_chunks_jsonl(Path(args.chunks_jsonl))
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
    llm_complete = build_minirag_llm(
        model,
        tokenizer,
        device,
        {"max_new_tokens": args.max_new_tokens, "do_sample": False},
    )
    adapter = MiniRAGAdapter(
        working_dir=Path(args.working_dir),
        embedding_model=args.embedding_model,
        llm_complete=llm_complete,
        chunk_records=chunk_records,
        top_k=args.rag_top_k,
        local_files_only=args.local_files_only,
        embedding_batch_size=args.embedding_batch_size,
        rebuild_index=args.rebuild_index,
        embedding_device=args.rag_device,
    )
    print(
        json.dumps(
            {
                "awq_backend": awq_backend,
                "load_timings": load_timings,
                "index_stats": adapter.get_index_stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
