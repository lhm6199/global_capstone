import argparse

import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from infer_awq import (
    cache_dequantized_weights,
    measure_step,
    normalize_hf_model_path,
    pick_awq_backend,
    pick_device,
    print_timings,
    replace_decoder_linears_with_awq,
)


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


class StreamingTextStreamer:
    def __init__(self, tokenizer, show_special_tokens=False):
        self.tokenizer = tokenizer
        self.show_special_tokens = show_special_tokens
        self._skipping_prompt = True

    def put(self, value):
        if self._skipping_prompt:
            self._skipping_prompt = False
            return

        token_ids = value.reshape(-1).tolist()
        for token_id in token_ids:
            text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=not self.show_special_tokens,
                clean_up_tokenization_spaces=False,
            )
            print(text, end="", flush=True)

    def end(self):
        pass


def load_awq_model(args):
    timings = {}
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = args.device or pick_device()
    awq_backend = pick_awq_backend(args.awq_backend, device)
    if awq_backend == "kernel" and not str(device).startswith("cuda"):
        raise ValueError("--awq_backend kernel requires a CUDA device.")

    model_path = normalize_hf_model_path(args.model_path)

    with measure_step(timings, "load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
        )

    with measure_step(timings, "load_config"):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    with measure_step(timings, "init_empty_model"):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

    with measure_step(timings, "replace_linear_modules"):
        replace_decoder_linears_with_awq(
            model,
            args.w_bit,
            args.q_group_size,
            awq_backend,
        )
        model.tie_weights()

    with measure_step(timings, "load_checkpoint_to_device", device):
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map={"": device},
            offload_state_dict=True,
        )

    with measure_step(timings, "model_to_eval", device):
        model = model.to(device).eval()

    if args.cache_dequantized_weights:
        if awq_backend == "torch_fallback":
            with measure_step(timings, "cache_dequantized_weights", device):
                cache_dequantized_weights(model)
        else:
            print("--cache_dequantized_weights is ignored with --awq_backend kernel.")

    return model, tokenizer, device, awq_backend, timings


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
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


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
    args = parser.parse_args(argv)

    model, tokenizer, device, awq_backend, timings = load_awq_model(args)
    if not args.no_timing:
        print(f"awq_backend: {awq_backend}")
        print_timings(timings)

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
        messages = trim_history(messages, args.history_max_turns)

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
