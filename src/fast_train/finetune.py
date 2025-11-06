#!/usr/bin/env python
# -*- coding: utf-8 -*-
__all__ = ["finetune_lora", "FinetuneParams"]
"""
LoRA finetuning with optional QLoRA via bitsandbytes.

Key features:
- 4-bit / 8-bit loading via bitsandbytes, safe for single-GPU and DDP.
- Explicit device_map for quantized models so *load device == train device*.
- Standard DDP through Hugging Face Trainer (no FSDP here).
- Safe "merge-and-save" path to bake LoRA adapters into base weights.
- Train/eval split pulled from a single JSONL file using HF datasets.
- If --merge_and_save is given and an existing runs/.../checkpoint-xxx is found
  under --out, the script skips training and merges that checkpoint.

Usage (single GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen3_lora.py \
        --model Qwen/Qwen3-1.7B \
        --data  data/common-v2.chat.jsonl \
        --out   runs/qwen3-1.7b-lora \
        --epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --merge_and_save

Usage (multi-GPU DDP):
    accelerate launch --num_processes=8 scripts/train_qwen3_lora.py \
        --model Qwen/Qwen3-1.7B \
        --data  data/common-v2.chat.jsonl \
        --out   runs/qwen3-1.7b-lora \
        --epochs 2 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --merge_and_save
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional, List, Dict, Any, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, DatasetDict
import importlib.util

try:
    # BitsAndBytesConfig is optional; only present if bitsandbytes is installed
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # type: ignore

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft import PeftModel


# ----------------------------
# CLI argument structure
# ----------------------------
@dataclass
class _ScriptArgs:
    model_name_or_path: str
    data_path: str
    output_dir: str
    max_seq_len: int
    lr: float
    epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    use_4bit: bool
    use_8bit: bool
    target_modules: Optional[List[str]]
    truncate_side: str
    merge_and_save: bool
    push_to_hub: bool
    load_best_model_at_end: bool


# ----------------------------
# Training params dataclass (public API)
# ----------------------------
@dataclass
class FinetuneParams:
    # Sequence/optimization
    max_seq_len: int = 4096
    lr: float = 2e-4
    epochs: float = 3.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    logging_steps: int = 20
    eval_steps: int = 200
    save_steps: int = 200

    # LoRA hyperparams
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv_proj",
            "w1",
            "w2",
            "w3",
        ]
    )

    # Truncation policy for long sequences
    truncate_side: str = "right"

    # Quantization toggles
    use_4bit: bool = True
    use_8bit: bool = False

    # Save/Hub options
    merge_and_save: bool = False
    push_to_hub: bool = False
    load_best_model_at_end: bool = False

    # Eval split + randomness
    eval_ratio: float = 0.02
    seed: int = 42

    # Required for merging when a preloaded model object is passed
    base_model_name_or_path: Optional[str] = None


def _parse_args() -> "_ScriptArgs":
    p = argparse.ArgumentParser()
    # Core model/data/output
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", dest="model_name_or_path")
    p.add_argument("--data", default="data/common-v2.chat.jsonl", dest="data_path")
    p.add_argument("--out", default="runs/qwen3-1.7b-lora", dest="output_dir")

    # Sequence/optimization
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)

    # LoRA hyperparams
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Quantization toggles
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit QLoRA.")
    p.add_argument("--use_8bit", action="store_true", help="Enable 8-bit quantization.")

    # Save options
    p.add_argument(
        "--merge_and_save",
        action="store_true",
        help="Merge LoRA into base weights at the end (or from last checkpoint if available).",
    )

    # LoRA target modules
    p.add_argument(
        "--target_modules",
        nargs="*",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "qkv_proj",
            "w1",
            "w2",
            "w3",
        ],
        help="Module names to apply LoRA to; depends on the base model architecture.",
    )

    # Truncation policy for long sequences
    p.add_argument("--truncate_side", choices=["left", "right"], default="right")

    # Hub options
    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--load_best_model_at_end", action="store_true")

    a = p.parse_args()
    return _ScriptArgs(
        model_name_or_path=a.model_name_or_path,
        data_path=a.data_path,
        output_dir=a.output_dir,
        max_seq_len=a.max_seq_len,
        lr=a.lr,
        epochs=a.epochs,
        per_device_train_batch_size=a.per_device_train_batch_size,
        per_device_eval_batch_size=a.per_device_eval_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        warmup_ratio=a.warmup_ratio,
        logging_steps=a.logging_steps,
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        lora_r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        use_4bit=(not a.no_4bit),
        use_8bit=a.use_8bit,
        target_modules=a.target_modules,
        truncate_side=a.truncate_side,
        merge_and_save=a.merge_and_save,
        push_to_hub=a.push_to_hub,
        load_best_model_at_end=a.load_best_model_at_end,
    )


# ----------------------------
# Bitsandbytes config (optional)
# ----------------------------
def _get_bnb_config(use_4bit: bool, use_8bit: bool):
    """
    Create bitsandbytes config if available and requested.
    Returns None if bnb is not installed or user didn't choose 4/8bit.
    """
    has_bnb = importlib.util.find_spec("bitsandbytes") is not None
    if BitsAndBytesConfig is None or not has_bnb:
        return None

    if use_4bit and not use_8bit:
        # 4-bit QLoRA (NF4 + double quant is a common choice)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if use_8bit and not use_4bit:
        # 8-bit weight-only
        return BitsAndBytesConfig(load_in_8bit=True)

    # If both flags are off (or both on), default to no quantization here
    return None


# ----------------------------
# Tokenizer helper
# ----------------------------
def _build_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(
        name_or_path, use_fast=False, trust_remote_code=True
    )
    # Ensure pad token exists and padding side is consistent with collator
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


# ----------------------------
# Device-map logic for bnb (CRITICAL FIX)
# ----------------------------
def _resolve_device_map_for_bnb():
    """
    For 4/8-bit (bitsandbytes) training, the load device MUST equal the train device.
    In DDP, each rank should fully own the model on its single GPU.
    We therefore avoid device_map="auto" for quantized models and instead pin to LOCAL_RANK.

    Returns:
        dict suitable for HF `device_map` when using bnb, like {"": 0}, {"": 1}, ...
    """
    # CUDA path (typical and recommended for bnb)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return {"": local_rank}

    # If you truly use Intel XPU and compatible bnb build, uncomment:
    # if hasattr(torch, "xpu") and torch.xpu.is_available():
    #     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    #     torch.xpu.set_device(local_rank)
    #     return {"": local_rank}

    raise RuntimeError(
        "4/8-bit training with bitsandbytes requires a supported accelerator (CUDA). "
        "Loading a quantized model on CPU/MPS then training on another device triggers the exact error you saw."
    )


# ----------------------------
# Model loader
# ----------------------------
def _load_model(name_or_path: str, bnb_cfg):
    """
    Load the base model with correct device mapping.
    - For quantized (bnb) models: DO NOT use device_map='auto'. Pin to the (local) training device.
    - For non-quantized models: device_map='auto' is fine.
    """
    if bnb_cfg is not None:
        device_map = _resolve_device_map_for_bnb()
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,  # <--- CRITICAL for bnb
            low_cpu_mem_usage=True,
            quantization_config=bnb_cfg,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    # Disable KV cache during training
    model.config.use_cache = False
    return model


# ----------------------------
# Data processing
# ----------------------------
def _to_features(
    examples: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_len: int,
    truncate_side: str,
):
    """
    Convert chat-format records into tokenized features.
    Expected input example:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]}
    We construct labels such that the prompt tokens are masked (-100) and only
    the assistant portion contributes to loss.
    """
    input_ids_batch, attn_batch, labels_batch = [], [], []
    for msgs in examples["messages"]:
        msgs = list(msgs)
        if len(msgs) < 2:
            continue

        # Build a "prompt only" text (no assistant answer) and a "full" text
        msgs_prompt = [m for m in msgs if m.get("role") != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # Mask prompt tokens; only assistant tokens produce loss
        if len(full_ids) < len(prompt_ids):
            # Safety fallback
            ids = full_ids
            labels = ids.copy()
        else:
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            ids = full_ids

        attn = [1] * len(ids)

        # Truncate to max length (left or right)
        if len(ids) > max_len:
            overflow = len(ids) - max_len
            if truncate_side == "left":
                ids = ids[overflow:]
                attn = attn[overflow:]
                labels = labels[overflow:]
            else:
                ids = ids[:-overflow]
                attn = attn[:-overflow]
                labels = labels[:-overflow]

        input_ids_batch.append(ids)
        attn_batch.append(attn)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attn_batch,
        "labels": labels_batch,
    }


# ----------------------------
# Checkpoint helpers (for merge-and-save shortcut)
# ----------------------------
def _find_checkpoints(output_dir: str) -> List[Tuple[int, str]]:
    if not os.path.isdir(output_dir):
        return []
    cks: List[Tuple[int, str]] = []
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[-1])
                cks.append((step, path))
            except Exception:
                pass
    cks.sort(key=lambda x: x[0])
    return cks


def _get_latest_checkpoint(output_dir: str) -> Optional[str]:
    cks = _find_checkpoints(output_dir)
    return cks[-1][1] if cks else None


# ----------------------------
# Merge adapters into base weights
# ----------------------------
def _merge_from_checkpoint_or_raise(
    base_model_name: str,
    tokenizer,
    checkpoint_dir: str,
    out_dir: str,
) -> None:
    print(f"[INFO] Using existing checkpoint for merge: {checkpoint_dir}")

    # Load base model in FP16 on a GPU (device_map='auto' is fine for non-bnb here)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Attach LoRA adapters then merge
    peft_model = PeftModel.from_pretrained(
        base,
        checkpoint_dir,
        is_trainable=False,
    ).to(dtype=torch.float16)

    merged = peft_model.merge_and_unload()

    merged_dir = os.path.join(out_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"[OK] Merged weights saved to: {merged_dir}")


# ----------------------------
# Public API: single-call LoRA finetune
# ----------------------------
def _as_finetune_params(params: Union["FinetuneParams", Dict[str, Any]]) -> "FinetuneParams":
    """Coerce a dict or FinetuneParams into a FinetuneParams instance.
    Unknown keys in dict are ignored; defaults fill any missing keys.
    """
    if isinstance(params, FinetuneParams):
        return params
    if isinstance(params, dict):
        valid = {f.name for f in dataclass_fields(FinetuneParams)}
        filtered = {k: v for k, v in params.items() if k in valid}
        return FinetuneParams(**filtered)  # type: ignore[arg-type]
    raise TypeError("params must be FinetuneParams or dict")

def finetune_lora(
    model: Union[str, Any],
    data: Union[str, Dataset, DatasetDict, List[Dict[str, Any]]],
    output_dir: str,
    params: Union[FinetuneParams, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Start a LoRA (optionally QLoRA) fine-tune in one call.

    Args:
        model: Base model identifier (e.g. "Qwen/Qwen3-1.7B") or a loaded
            `transformers.PreTrainedModel`. If a model object is provided, 4/8‑bit
            quantization is not applied automatically; to use QLoRA, pass a string
            so it can be loaded with bitsandbytes.
        data: One of
            - path to a JSONL file with a "messages" column in chat format
            - a `datasets.Dataset` or `DatasetDict` with a "messages" column
            - a Python list of dict rows (each row has a "messages" field)
        output_dir: Directory to save checkpoints/adapters and (optionally) merged weights.
        params: FinetuneParams or a dict with training/LoRA/quantization/save/hub settings.

    Returns:
        Dict with evaluation metrics and a few paths of interest.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    params = _as_finetune_params(params)

    # Determine tokenizer source and bnb configuration
    if isinstance(model, str):
        model_name = model
        tokenizer = _build_tokenizer(model_name)
        bnb_cfg = _get_bnb_config(params.use_4bit, params.use_8bit)
        model_obj = _load_model(model_name, bnb_cfg)
        if bnb_cfg is not None:
            model_obj = prepare_model_for_kbit_training(model_obj)
        use_bnb_optim = bnb_cfg is not None
    else:
        # Preloaded model object
        model_obj = model
        model_name = getattr(model_obj, "name_or_path", None) or params.base_model_name_or_path
        if not model_name:
            raise ValueError(
                "When passing a model object, provide `params.base_model_name_or_path` to build the tokenizer and for merging."
            )
        tokenizer = _build_tokenizer(model_name)
        bnb_cfg = None  # cannot auto-quantize an already loaded model here
        use_bnb_optim = False

    # Configure special tokens and save early configs
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    model_obj.config.use_cache = False
    model_obj.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model_obj, "generation_config") and model_obj.generation_config is not None:
        model_obj.generation_config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.bos_token_id is not None:
            model_obj.generation_config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.bos_token_id is not None:
        model_obj.config.bos_token_id = tokenizer.bos_token_id

    model_obj.config.to_json_file(os.path.join(output_dir, "config.json"))
    if hasattr(model_obj, "generation_config") and model_obj.generation_config is not None:
        model_obj.generation_config.to_json_file(
            os.path.join(output_dir, "generation_config.json")
        )

    # Build LoRA peft model
    peft_cfg = LoraConfig(
        r=params.lora_r,
        lora_alpha=params.lora_alpha,
        lora_dropout=params.lora_dropout,
        target_modules=params.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model_obj = get_peft_model(model_obj, peft_cfg)
    model_obj.print_trainable_parameters()

    # Load/prepare dataset
    if isinstance(data, str):
        ds_train_full = load_dataset("json", data_files=data, split="train")
    elif isinstance(data, DatasetDict):
        # expect keys like {"train": ..., "validation": ...}
        if "train" in data:
            ds_train_full = data["train"]
        else:
            raise ValueError("DatasetDict must contain a 'train' split.")
    elif isinstance(data, Dataset):
        ds_train_full = data
    elif isinstance(data, list):
        from datasets import Dataset as HFDataset

        ds_train_full = HFDataset.from_list(data)
    else:
        raise TypeError("Unsupported `data` type. Provide path, Dataset, DatasetDict, or list of rows.")

    n = len(ds_train_full)
    eval_size = max(1, int(params.eval_ratio * n)) if n > 0 else 1
    ds_train_full = ds_train_full.shuffle(seed=params.seed)
    ds_train = ds_train_full.select(range(eval_size, n)) if n > 1 else ds_train_full
    ds_eval = ds_train_full.select(range(0, eval_size)) if n > 1 else ds_train_full.select(range(0, 0))

    def _map_fn(batch):
        return _to_features(batch, tokenizer, params.max_seq_len, params.truncate_side)

    ds_train_tok = ds_train.map(_map_fn, batched=True, remove_columns=ds_train.column_names)
    ds_eval_tok = (
        ds_eval.map(_map_fn, batched=True, remove_columns=ds_eval.column_names)
        if len(ds_eval) > 0
        else None
    )

    # Collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, padding=True)

    # Precision
    has_cuda = torch.cuda.is_available()
    is_bf16 = bool(has_cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_fp16 = bool(has_cuda and not is_bf16)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=params.epochs,
        per_device_train_batch_size=params.per_device_train_batch_size,
        per_device_eval_batch_size=params.per_device_eval_batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        learning_rate=params.lr,
        warmup_ratio=params.warmup_ratio,
        logging_steps=params.logging_steps,
        eval_steps=params.eval_steps,
        save_steps=params.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        bf16=is_bf16,
        fp16=use_fp16,
        report_to=["none"],
        optim="paged_adamw_8bit" if use_bnb_optim else "adamw_torch",
        ddp_find_unused_parameters=False,
        push_to_hub=params.push_to_hub,
        load_best_model_at_end=params.load_best_model_at_end,
    )

    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate() if ds_eval_tok is not None else {}
    if "eval_loss" in metrics:
        try:
            metrics["eval_ppl"] = math.exp(metrics["eval_loss"])
        except Exception:
            pass

    trainer.save_state()
    trainer.save_model(output_dir)

    # Optional merge
    if params.merge_and_save:
        try:
            ckpt = _get_latest_checkpoint(output_dir)
            if ckpt is None:
                root_adapter_cfg = os.path.join(output_dir, "adapter_config.json")
                if os.path.exists(root_adapter_cfg):
                    ckpt = output_dir
            if ckpt is None:
                raise RuntimeError(
                    "No LoRA checkpoint found for merging. Ensure a checkpoint-* exists or adapters are in run root."
                )
            base_for_merge = params.base_model_name_or_path or model_name
            if not base_for_merge:
                raise ValueError(
                    "`params.base_model_name_or_path` is required to merge adapters when `model` is a loaded object."
                )
            _merge_from_checkpoint_or_raise(
                base_model_name=base_for_merge,
                tokenizer=tokenizer,
                checkpoint_dir=ckpt,
                out_dir=output_dir,
            )
        except Exception as e:
            print(f"[ERROR] Merge failed: {e}")

    # Return a compact result for programmatic use
    return {
        "metrics": metrics,
        "output_dir": output_dir,
        "merged_dir": os.path.join(output_dir, "merged") if merge_and_save else None,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    args = _parse_args()
    params = FinetuneParams(
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        truncate_side=args.truncate_side,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
        merge_and_save=args.merge_and_save,
        push_to_hub=args.push_to_hub,
        load_best_model_at_end=args.load_best_model_at_end,
        eval_ratio=0.02,
        seed=42,
        base_model_name_or_path=args.model_name_or_path,
    )

    result = finetune_lora(
        model=args.model_name_or_path,
        data=args.data_path,
        output_dir=args.output_dir,
        params=params,
    )

    # Print metrics in a JSON-like line for easy scraping
    metrics = result.get("metrics", {})
    try:
        print({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})
    except Exception:
        print({"info": "Training completed."})


if __name__ == "__main__":
    main()
