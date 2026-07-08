import argparse
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import awq_inference_engine


FALLBACK_OP_PROFILE = False
FALLBACK_OP_TIMINGS = {
    "fallback_dequantize_weight": 0.0,
    "fallback_linear": 0.0,
    "fallback_forward": 0.0,
    "fallback_calls": 0,
}


def sync_device(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


@contextmanager
def measure_step(timings, name, device=None):
    if device is not None:
        sync_device(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device is not None:
            sync_device(device)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start


def print_timings(timings, generated_token_count=None):
    print("\n[timing]")
    for name, elapsed in timings.items():
        print(f"{name}: {elapsed:.4f}s")

    if generated_token_count is not None and timings.get("generate"):
        tokens_per_sec = generated_token_count / timings["generate"]
        print(f"generated_tokens: {generated_token_count}")
        print(f"tokens_per_second: {tokens_per_sec:.2f}")

    if FALLBACK_OP_TIMINGS["fallback_calls"]:
        calls = FALLBACK_OP_TIMINGS["fallback_calls"]
        print(f"fallback_calls: {calls}")
        for name in ("fallback_dequantize_weight", "fallback_linear", "fallback_forward"):
            elapsed = FALLBACK_OP_TIMINGS[name]
            print(f"{name}: {elapsed:.4f}s")
            print(f"{name}_per_call: {elapsed / calls:.6f}s")


def profile_fallback_op(name, fn, device):
    sync_device(device)
    start = time.perf_counter()
    result = fn()
    sync_device(device)
    FALLBACK_OP_TIMINGS[name] += time.perf_counter() - start
    return result


def make_divisible(value, divisor):
    return (value + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError(f"Unsupported group size: {group_size}")

    base_width = make_divisible(in_features // group_size, pack_num)
    return make_divisible(base_width, size_multiplier) * size_multiplier


class WQLinearNoKernel(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dtype, device):
        super().__init__()
        if w_bit != 4:
            raise NotImplementedError("Only packed 4-bit AWQ checkpoints are supported.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.interleave = 4

        pack_num = 32 // self.w_bit
        int16_pack_num = 16 // self.w_bit
        zeros_width = calculate_zeros_width(in_features, self.group_size) * pack_num

        self.register_buffer(
            "qweight",
            torch.empty(
                out_features // self.interleave,
                in_features // int16_pack_num * self.interleave,
                dtype=torch.int16,
                device=device,
            ),
        )
        self.register_buffer(
            "scales",
            torch.empty(zeros_width, out_features, dtype=dtype, device=device),
        )
        self.register_buffer(
            "scaled_zeros",
            torch.empty(zeros_width, out_features, dtype=dtype, device=device),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size):
        return cls(
            w_bit=w_bit,
            group_size=group_size,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )

    def dequantize_weight(self):
        qweight = unpack_awq_int4_weight(
            self.qweight,
            out_features=self.out_features,
            in_features=self.in_features,
        ).to(self.scales.dtype)

        group_ids = torch.arange(self.in_features, device=self.scales.device)
        group_ids = torch.div(group_ids, self.group_size, rounding_mode="floor")
        scales = self.scales.index_select(0, group_ids).transpose(0, 1).contiguous()
        scaled_zeros = self.scaled_zeros.index_select(0, group_ids).transpose(0, 1)
        return qweight * scales + scaled_zeros

    @torch.no_grad()
    def forward(self, x):
        if not FALLBACK_OP_PROFILE:
            weight = getattr(self, "_dequantized_weight", None)
            if weight is None:
                weight = self.dequantize_weight()
            return F.linear(x, weight.to(dtype=x.dtype), self.bias)

        device = x.device
        FALLBACK_OP_TIMINGS["fallback_calls"] += 1
        sync_device(device)
        forward_start = time.perf_counter()

        weight = getattr(self, "_dequantized_weight", None)
        if weight is None:
            weight = profile_fallback_op(
                "fallback_dequantize_weight",
                self.dequantize_weight,
                device,
            )

        out = profile_fallback_op(
            "fallback_linear",
            lambda: F.linear(x, weight.to(dtype=x.dtype), self.bias),
            device,
        )
        sync_device(device)
        FALLBACK_OP_TIMINGS["fallback_forward"] += time.perf_counter() - forward_start
        return out


def unpack_awq_int4_weight(qweight, out_features, in_features):
    if in_features % 64 != 0:
        raise ValueError("AWQ packed INT4 weights expect in_features divisible by 64.")
    if out_features % 4 != 0:
        raise ValueError("AWQ packed INT4 weights expect out_features divisible by 4.")

    packed = qweight.to(torch.int32) & 0xFFFF
    unpacked = torch.stack(
        [(packed >> shift) & 0xF for shift in (0, 4, 8, 12)],
        dim=-1,
    )

    unpacked = unpacked.reshape(out_features // 4, in_features // 64, 64, 4)
    unpacked = unpacked.reshape(out_features // 4, in_features // 64, 4, 64)
    unpacked = unpacked.transpose(1, 2).reshape(out_features, in_features)

    unpacked = unpacked.reshape(out_features, in_features // 32, 4, 2, 4)
    unpacked = unpacked.transpose(3, 4).reshape(out_features, in_features // 32, 4, 8)
    unpacked = unpacked.reshape(out_features, in_features // 32, 4, 4, 2)
    return unpacked.transpose(2, 3).reshape(out_features, in_features)


def set_module_by_name(root, name, module):
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def get_decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    raise NotImplementedError(f"Unsupported model class: {model.__class__.__name__}")


def get_kernel_awq_linear():
    from awq.quantize.qmodule import WQLinear, _get_awq_inference_engine

    _get_awq_inference_engine()
    return WQLinear


def make_awq_linear(module, w_bit, q_group_size, backend):
    if backend == "kernel":
        WQLinear = get_kernel_awq_linear()
        return WQLinear.from_linear(module, w_bit, q_group_size, init_only=True)
    return WQLinearNoKernel.from_linear(module, w_bit, q_group_size)


def replace_decoder_linears_with_awq(model, w_bit, q_group_size, backend):
    for layer in get_decoder_layers(model):
        linears = [
            (name, module)
            for name, module in layer.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, module in linears:
            set_module_by_name(
                layer,
                name,
                make_awq_linear(module, w_bit, q_group_size, backend),
            )


@torch.no_grad()
def cache_dequantized_weights(model):
    for module in model.modules():
        if isinstance(module, WQLinearNoKernel):
            module.register_buffer(
                "_dequantized_weight",
                module.dequantize_weight(),
                persistent=False,
            )


def normalize_hf_model_path(model_path):
    prefix = "https://huggingface.co/"
    if model_path.startswith(prefix):
        return model_path[len(prefix) :].rstrip("/")
    return model_path


def pick_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def pick_awq_backend(requested_backend, device):
    if requested_backend != "auto":
        return requested_backend
    if str(device).startswith("cuda"):
        return "kernel"
    return "torch_fallback"


def build_inputs(tokenizer, prompt, device):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def main(argv=None, default_awq_backend="auto"):
    parser = argparse.ArgumentParser(
        description="Run packed AWQ INT4 inference."
    )
    parser.add_argument("--model_path", default="Qwen/Qwen3-4B")
    parser.add_argument("--load_quant", required=True)
    parser.add_argument("--prompt", default="Explain AWQ quantization in one paragraph.")
    parser.add_argument("--w_bit", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--awq_backend",
        choices=["auto", "kernel", "torch_fallback"],
        default=default_awq_backend,
        help=(
            "AWQ execution backend. 'kernel' uses awq_inference_engine packed "
            "INT4 CUDA kernels; 'torch_fallback' dequantizes to dense weights."
        ),
    )
    parser.add_argument("--cache_dequantized_weights", action="store_true")
    parser.add_argument(
        "--no_timing",
        action="store_true",
        help="Disable stage timing output.",
    )
    parser.add_argument(
        "--profile_fallback_ops",
        action="store_true",
        help=(
            "When using torch_fallback, synchronize and print accumulated "
            "dequantize/F.linear timings. This adds measurement overhead."
        ),
    )
    args = parser.parse_args(argv)

    global FALLBACK_OP_PROFILE
    FALLBACK_OP_PROFILE = args.profile_fallback_ops
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
        replace_decoder_linears_with_awq(model, args.w_bit, args.q_group_size, awq_backend)
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

    with measure_step(timings, "build_inputs", device):
        inputs = build_inputs(tokenizer, args.prompt, device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

    with measure_step(timings, "generate", device):
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    with measure_step(timings, "decode"):
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    print(generated_text)
    if not args.no_timing:
        print(f"awq_backend: {awq_backend}")
        print_timings(timings, generated_token_count=generated_ids.numel())


if __name__ == "__main__":
    main()
