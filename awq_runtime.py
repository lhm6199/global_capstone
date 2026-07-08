from __future__ import annotations

from types import SimpleNamespace

import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from infer_awq import (
    build_inputs,
    cache_dequantized_weights,
    measure_step,
    normalize_hf_model_path,
    pick_awq_backend,
    pick_device,
    replace_decoder_linears_with_awq,
)


def _config_value(config, name, default=None):
    return getattr(config, name, default)


def make_runtime_config(**kwargs):
    return SimpleNamespace(**kwargs)


def load_awq_runtime(config):
    timings = {}
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[_config_value(config, "dtype", "float16")]
    device = _config_value(config, "device") or pick_device()
    awq_backend = pick_awq_backend(_config_value(config, "awq_backend", "auto"), device)
    if awq_backend == "kernel" and not str(device).startswith("cuda"):
        raise ValueError("--awq_backend kernel requires a CUDA device.")

    model_path = normalize_hf_model_path(_config_value(config, "model_path"))
    local_files_only = bool(_config_value(config, "local_files_only", False))

    with measure_step(timings, "load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

    with measure_step(timings, "load_config"):
        hf_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

    with measure_step(timings, "init_empty_model"):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=hf_config,
                dtype=dtype,
                trust_remote_code=True,
            )

    with measure_step(timings, "replace_linear_modules"):
        replace_decoder_linears_with_awq(
            model,
            _config_value(config, "w_bit", 4),
            _config_value(config, "q_group_size", 128),
            awq_backend,
        )
        model.tie_weights()

    with measure_step(timings, "load_checkpoint_to_device", device):
        load_checkpoint_in_model(
            model,
            checkpoint=_config_value(config, "load_quant"),
            device_map={"": device},
            offload_state_dict=True,
        )

    with measure_step(timings, "model_to_eval", device):
        model = model.to(device).eval()

    if _config_value(config, "cache_dequantized_weights", False):
        if awq_backend == "torch_fallback":
            with measure_step(timings, "cache_dequantized_weights", device):
                cache_dequantized_weights(model)
        else:
            print("--cache_dequantized_weights is ignored with --awq_backend kernel.")

    return model, tokenizer, device, awq_backend, timings


def generate_from_prompt(model, tokenizer, device, prompt, generation_config):
    timings = {}
    with measure_step(timings, "build_inputs", device):
        inputs = build_inputs(tokenizer, prompt, device)

    generate_kwargs = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "do_sample": generation_config.get("do_sample", False),
        "pad_token_id": generation_config.get("pad_token_id", tokenizer.eos_token_id),
    }
    if generation_config.get("do_sample", False):
        generate_kwargs.update(
            {
                "temperature": generation_config.get("temperature", 0.7),
                "top_p": generation_config.get("top_p", 0.95),
            }
        )

    with measure_step(timings, "generate", device):
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generate_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    with measure_step(timings, "decode"):
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    answer_tokens = int(generated_ids.numel())
    generate_s = timings.get("generate", 0.0)
    timings["tokens_per_second"] = (
        answer_tokens / generate_s if answer_tokens > 0 and generate_s > 0 else 0.0
    )
    return {
        "text": text,
        "answer_tokens": answer_tokens,
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "timings": timings,
    }
