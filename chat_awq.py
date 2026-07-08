import argparse
import time

import torch

from awq_runtime import generate_from_prompt, load_awq_runtime, make_runtime_config
from infer_awq import (
    measure_step,
    print_timings,
)
from rag import RagChatTurn, RagPromptBuilder, RagRetriever, build_rag_metrics, print_metrics


def build_chat_inputs(tokenizer, messages, device):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        text = f"{text}\nassistant:"
    return tokenizer(text, return_tensors="pt").to(device)


def trim_history(messages, max_turns):
    if max_turns <= 0:
        return messages

    system_messages = [message for message in messages if message["role"] == "system"]
    chat_messages = [message for message in messages if message["role"] != "system"]
    return system_messages + chat_messages[-max_turns * 2 :]


def build_rag_pipeline(args):
    if not args.rag_index_dir:
        return None

    retriever = RagRetriever.from_index_dir(
        args.rag_index_dir,
        args.embedding_model,
        batch_size=args.rag_embedding_batch_size,
        local_files_only=args.rag_local_files_only,
        device=args.rag_device,
    )
    prompt_builder = RagPromptBuilder.from_file(args.rag_prompt_template)
    return RagChatTurn(retriever=retriever, prompt_builder=prompt_builder, top_k=args.rag_top_k)


def load_awq_model(args):
    config = make_runtime_config(
        model_path=args.model_path,
        load_quant=args.load_quant,
        w_bit=args.w_bit,
        q_group_size=args.q_group_size,
        dtype=args.dtype,
        device=args.device,
        awq_backend=args.awq_backend,
        cache_dequantized_weights=args.cache_dequantized_weights,
        local_files_only=getattr(args, "local_files_only", False),
    )
    return load_awq_runtime(config)


def generate_reply(model, tokenizer, messages, device, args):
    inputs = build_chat_inputs(tokenizer, messages, device)
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
    if args.show_tokens:
        generate_kwargs["streamer"] = StreamingTextStreamer(
            tokenizer,
            show_special_tokens=args.show_special_tokens,
        )

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return {
        "reply": tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
        "answer_tokens": int(generated_ids.numel()),
    }


def generate_reply_from_prompt(model, tokenizer, prompt, device, args):
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_config.update({"temperature": args.temperature, "top_p": args.top_p})
    result = generate_from_prompt(model, tokenizer, device, prompt, generation_config)
    return {
        "reply": result["text"],
        "answer_tokens": result["answer_tokens"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Chat with a packed AWQ INT4 causal language model."
    )
    parser.add_argument("--model_path", default="qwen3-4b-awq-runtime")
    parser.add_argument("--load_quant", default="model/qwen3-4b-w4-g128-awq-v2.pt")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--show_tokens",
        action="store_true",
        help="Stream generated text token by token while the model is running.",
    )
    parser.add_argument(
        "--show_special_tokens",
        action="store_true",
        help="Show special tokens in --show_tokens streaming output instead of hiding them.",
    )
    parser.add_argument(
        "--awq_backend",
        choices=["auto", "kernel", "torch_fallback"],
        default="auto",
        help=(
            "AWQ execution backend. 'kernel' uses awq_inference_engine packed "
            "INT4 CUDA kernels; 'torch_fallback' dequantizes to dense weights."
        ),
    )
    parser.add_argument("--cache_dequantized_weights", action="store_true")
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system message inserted at the start of the chat.",
    )
    parser.add_argument(
        "--history_max_turns",
        type=int,
        default=8,
        help="Keep this many recent user/assistant turns. Use 0 to keep all history.",
    )
    parser.add_argument(
        "--no_timing",
        action="store_true",
        help="Disable model loading timing output.",
    )
    parser.add_argument(
        "--rag_index_dir",
        default=None,
        help="Enable RAG by loading chunks.jsonl and faiss.index from this directory.",
    )
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument(
        "--rag_prompt_template",
        default=None,
        help="Optional prompt template file. Defaults to rag/templates/default_rag_prompt.txt.",
    )
    parser.add_argument(
        "--embedding_model",
        default="BAAI/bge-base-en-v1.5",
        help="Sentence-transformers model used for retrieval query embedding.",
    )
    parser.add_argument(
        "--rag_embedding_batch_size",
        type=int,
        default=32,
        help="Batch size for sentence-transformers encoding.",
    )
    parser.add_argument(
        "--rag_local_files_only",
        action="store_true",
        help="Load the embedding model from local cache only.",
    )
    parser.add_argument(
        "--rag_device",
        default=None,
        help="Optional device override for retrieval embeddings, e.g. cpu or cuda.",
    )
    args = parser.parse_args(argv)

    model, tokenizer, device, awq_backend, timings = load_awq_model(args)
    rag_chat = build_rag_pipeline(args)
    if not args.no_timing:
        print(f"awq_backend: {awq_backend}")
        print_timings(timings)
    if rag_chat and args.system_prompt:
        print("RAG mode ignores --system_prompt and prior chat history in the model prompt.")

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("\n채팅을 시작합니다. 종료: /exit 또는 /quit, 대화 초기화: /clear")
    while True:
        try:
            user_text = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("대화 기록을 초기화했습니다.")
            continue

        messages.append({"role": "user", "content": user_text})
        if not rag_chat:
            messages = trim_history(messages, args.history_max_turns)

        if rag_chat:
            rag_prepare_started = time.perf_counter()
            turn = rag_chat.prepare(user_text)
            rag_prepare_s = time.perf_counter() - rag_prepare_started
            generation = generate_reply_from_prompt(
                model,
                tokenizer,
                turn["prompt"],
                device,
                args,
            )
            reply = generation["reply"]
            if not args.no_timing:
                print_metrics(
                    build_rag_metrics(
                        tokenizer,
                        turn,
                        answer_tokens=generation["answer_tokens"],
                        rag_prepare_s=rag_prepare_s,
                    )
                )
        else:
            generation = generate_reply(model, tokenizer, messages, device, args)
            reply = generation["reply"]

        if args.show_tokens:
            print("\nAssistant: ", end="", flush=True)
        reply = generate_reply(model, tokenizer, messages, device, args)
        messages.append({"role": "assistant", "content": reply})
        if args.show_tokens:
            print()
        else:
            print(f"\nAssistant: {reply}")


if __name__ == "__main__":
    main()
