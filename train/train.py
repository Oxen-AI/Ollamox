"""
Fine-tune a Gemma model using QLoRA with Hugging Face Transformers and TRL.

Follows the recipe at:
  https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

The script handles:
  1. Loading + preparing a conversational fine-tuning dataset
  2. 4-bit QLoRA quantisation via BitsAndBytes
  3. LoRA adapter training with the TRL SFTTrainer
  4. Optional adapter merge into the base model for standalone inference
  5. Optional inference smoke-test on held-out examples

Usage:
    python train/train.py --model google/gemma-4-E2B

    python train/train.py \
        --model google/gemma-4-E4B \
        --dataset data/data.jsonl \
        --epochs 3 \
        --output train/output/gemma-finetune \
        --merge
"""

import argparse
import gc
import json
import os
from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _jsonl_row_to_messages(sample: dict) -> dict:
    """Convert a {"prompt": ..., "response": ...} row into the chat messages
    format that TRL's SFTTrainer expects.

    If the row already has a "messages" key it is passed through unchanged.
    An optional "system" key is included as the system message when present.
    """
    if "messages" in sample:
        return {"messages": sample["messages"]}

    messages = []
    if sample.get("system"):
        messages.append({"role": "system", "content": sample["system"]})
    messages.append({"role": "user", "content": sample["prompt"]})
    messages.append({"role": "assistant", "content": sample["response"]})
    return {"messages": messages}


def prepare_dataset(
    dataset_path: str, max_samples: int, test_size: float, seed: int
):
    """Load a local JSONL file (or HuggingFace dataset ID), convert to chat
    format, and split into train/eval."""
    path = Path(dataset_path)
    if path.is_file():
        print(f"Loading local dataset: {path}")
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        print(f"Loading HuggingFace dataset: {dataset_path}")
        ds = load_dataset(dataset_path, split="train")

    if max_samples and len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    else:
        ds = ds.shuffle(seed=seed)
    print(f"  {len(ds)} examples after subsampling")

    ds = ds.map(_jsonl_row_to_messages, remove_columns=ds.column_names, batched=False)
    splits = ds.train_test_split(test_size=test_size, seed=seed)

    print(f"  Train: {len(splits['train'])}  |  Eval: {len(splits['test'])}")
    return splits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str, tokenizer_id: str | None):
    """Load the base model with 4-bit quantisation and the instruction tokenizer."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )

    print(f"Loading model: {model_id}  (compute dtype: {compute_dtype})")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=compute_dtype,
        device_map={"": 0},
        quantization_config=bnb_config,
    )

    tok_id = tokenizer_id or _instruction_tokenizer(model_id)
    print(f"Loading tokenizer: {tok_id}")
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    return model, tokenizer, compute_dtype


def _instruction_tokenizer(model_id: str) -> str:
    """Derive the instruction-tuned tokenizer for a given base model id.

    google/gemma-4-E2B  -> google/gemma-4-E2B-it
    google/gemma-4-E4B  -> google/gemma-4-E4B-it
    anything else       -> model_id itself
    """
    lower = model_id.lower()
    if "gemma" in lower and not model_id.endswith("-it"):
        return model_id + "-it"
    return model_id


# ---------------------------------------------------------------------------
# LoRA / training config
# ---------------------------------------------------------------------------

def build_lora_config(
    r: int, alpha: int, dropout: float, target_modules: str
) -> LoraConfig:
    return LoraConfig(
        lora_alpha=alpha,
        lora_dropout=dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )


def build_training_args(
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    logging_steps: int,
    save_strategy: str,
    eval_strategy: str,
    compute_dtype: torch.dtype,
    push_to_hub: bool,
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    lr_scheduler_type: str,
) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir,
        max_length=max_length,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch_fused",
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        learning_rate=learning_rate,
        fp16=(compute_dtype == torch.float16),
        bf16=(compute_dtype == torch.bfloat16),
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        push_to_hub=push_to_hub,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        },
    )


# ---------------------------------------------------------------------------
# In-training diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_generate(model, tokenizer, input_ids, attention_mask=None,
                     max_new_tokens=256):
    """Greedy autoregressive generation using model.forward() directly.

    model.generate() delegates to the base Gemma4ForConditionalGeneration which
    calls prepare_inputs_for_generation → forward, bypassing the PeftModel
    wrapper.  This function calls model(**inputs) (the PEFT-wrapped forward)
    in a loop, matching the exact code path used during training.
    """
    device = input_ids.device
    generated = input_ids.clone()
    if attention_mask is None:
        attention_mask = torch.ones_like(generated)

    eos_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eid = tokenizer.eos_token_id
        if isinstance(eid, int):
            eos_ids.add(eid)
        elif eid is not None:
            eos_ids.update(eid)
    turn_end = tokenizer.convert_tokens_to_ids("<turn|>")
    if isinstance(turn_end, int):
        eos_ids.add(turn_end)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated,
            attention_mask=attention_mask,
            token_type_ids=torch.zeros_like(generated),
            mm_token_type_ids=torch.zeros_like(generated),
        )
        next_id = outputs.logits[:, -1, :].argmax(dim=-1)
        generated = torch.cat([generated, next_id.unsqueeze(-1)], dim=-1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
        ], dim=-1)
        if next_id.item() in eos_ids:
            break

    return generated


def _generate_samples(model, tokenizer, dataset, n_samples=2, max_new_tokens=32,
                      label="", show_tokens=False):
    """Run greedy generation on a few eval samples and print results.

    Uses forward_generate() which calls model(**inputs) in a loop -- the same
    code path as training -- instead of model.generate() which bypasses the
    PEFT wrapper on multimodal models.
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            messages = sample["messages"]
            context = [m for m in messages if m["role"] != "assistant"]
            expected = next(
                (m["content"] for m in messages if m["role"] == "assistant"), ""
            )

            inputs = tokenizer.apply_chat_template(
                context, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            if show_tokens:
                ids = inputs["input_ids"][0].tolist()
                decoded = tokenizer.convert_ids_to_tokens(ids)
                print(f"  Token IDs : {ids}")
                print(f"  Tokens    : {decoded}")

            output_ids = forward_generate(
                model, tokenizer,
                inputs["input_ids"], inputs.get("attention_mask"),
                max_new_tokens=max_new_tokens,
            )
            new_ids = output_ids[0][prompt_len:].tolist()
            generated = tokenizer.decode(new_ids, skip_special_tokens=False)

            print(f"\n  {label}Sample {i + 1}:")
            print(f"    Expected : {expected}")
            print(f"    Generated: {generated.strip()}")
            if show_tokens:
                print(f"    Gen IDs  : {new_ids}")

    if was_training:
        model.train()


class SampleGenerationCallback(TrainerCallback):
    """Print model predictions on a few eval examples every *every* steps."""

    def __init__(self, tokenizer, eval_dataset, every: int = 50, n_samples: int = 2):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.every = every
        self.n_samples = n_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every != 0 or model is None:
            return
        print(f"\n{'='*60}")
        print(f"  Generation check @ step {state.global_step}")
        print(f"{'='*60}")
        _generate_samples(
            model, self.tokenizer, self.eval_dataset,
            n_samples=self.n_samples, label=f"[step {state.global_step}] ",
        )
        print()


# ---------------------------------------------------------------------------
# Post-training: merge + inference
# ---------------------------------------------------------------------------

def _adapter_config_override(adapter_path: str) -> LoraConfig:
    """Build a LoraConfig with target_modules="all-linear" from a saved adapter.

    PeftModel.from_pretrained uses the *resolved* target_modules list persisted
    in adapter_config.json.  For Gemma4ForConditionalGeneration this list
    contains a mix of path formats (e.g. "language_model.layers.0.self_attn.q_proj"
    alongside "32.mlp.gate_proj") that fail PEFT's endswith-matching, leaving
    LoRA parameters as orphan tensors disconnected from the forward graph.

    Passing target_modules="all-linear" forces PEFT to re-discover linear
    modules at load time — the same strategy used during training — which
    correctly injects LoRA wrappers.
    """
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type=cfg.get("task_type", "CAUSAL_LM"),
        bias=cfg.get("bias", "none"),
    )


def _replace_submodule(root: torch.nn.Module, dotted_name: str, new: torch.nn.Module):
    """Replace a nested submodule by its dotted path."""
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _dequant_linear(module) -> torch.nn.Linear:
    """Dequantize a bnb Linear4bit into a regular nn.Linear (bfloat16)."""
    import bitsandbytes as bnb

    w = bnb.functional.dequantize_4bit(
        module.weight.data, module.weight.quant_state,
    ).to(torch.bfloat16)
    out = torch.nn.Linear(
        w.shape[1], w.shape[0],
        bias=module.bias is not None,
        device=w.device, dtype=w.dtype,
    )
    out.weight = torch.nn.Parameter(w)
    if module.bias is not None:
        out.bias = torch.nn.Parameter(module.bias.data.to(w.dtype))
    return out


def merge_adapter(model_id: str, adapter_dir: str, output_dir: str):
    """Merge a QLoRA adapter into the base model and save at full precision.

    The base model is loaded with the SAME 4-bit quantisation used during
    training so that ``dequant(W_4bit) + LoRA`` reproduces the training-time
    forward pass exactly.

    We bypass PEFT's ``merge_and_unload()`` because it re-quantises the merged
    weights back to 4-bit (destroying precision).  Instead we manually
    dequantise each base weight, add the LoRA delta in float, and store the
    result as a regular ``nn.Linear``.
    """
    import bitsandbytes as bnb

    print(f"\nMerging adapter into base model...")
    print(f"  Base model : {model_id}")
    print(f"  Adapter    : {adapter_dir}")
    print(f"  Output     : {output_dir}")

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )

    base = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=compute_dtype,
    )

    cfg = _adapter_config_override(adapter_dir)
    peft_model = get_peft_model(base, cfg)

    weights_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(weights_file):
        adapter_weights = load_safetensors(weights_file)
    else:
        adapter_weights = torch.load(
            os.path.join(adapter_dir, "adapter_model.bin"),
            map_location="cpu", weights_only=True,
        )
    set_peft_model_state_dict(peft_model, adapter_weights)

    # ---- Manual merge (avoids re-quantisation) ------------------------------
    lora_model = peft_model.base_model          # LoraModel wrapper
    print("  Merging LoRA weights (manual dequant, no re-quantisation)...")

    # Pass 1: merge each LoRA-wrapped module → float nn.Linear
    for name, module in list(lora_model.named_modules()):
        if not (hasattr(module, "get_base_layer") and hasattr(module, "lora_A")):
            continue
        if not module.lora_A:
            continue

        base_layer = module.get_base_layer()

        if hasattr(base_layer.weight, "quant_state"):
            w = bnb.functional.dequantize_4bit(
                base_layer.weight.data, base_layer.weight.quant_state,
            )
        else:
            w = base_layer.weight.data.clone()
        w = w.to(torch.bfloat16)

        for adapter_name in module.active_adapters:
            if adapter_name in module.lora_A:
                delta = module.get_delta_weight(adapter_name)
                w = w + delta.to(w.dtype)

        new_linear = torch.nn.Linear(
            w.shape[1], w.shape[0],
            bias=base_layer.bias is not None,
            device=w.device, dtype=w.dtype,
        )
        new_linear.weight = torch.nn.Parameter(w)
        if base_layer.bias is not None:
            new_linear.bias = torch.nn.Parameter(
                base_layer.bias.data.to(w.dtype),
            )
        _replace_submodule(lora_model, name, new_linear)

    merged = lora_model.model  # unwrap → Gemma4ForConditionalGeneration

    # Pass 2: dequantize remaining (non-LoRA) Linear4bit modules
    for name, module in list(merged.named_modules()):
        if isinstance(module, bnb.nn.Linear4bit):
            _replace_submodule(merged, name, _dequant_linear(module))

    # Remove quantization metadata so the saved model loads as plain bfloat16
    for attr in ("quantization_config", "hf_quantizer", "_pre_quantization_dtype"):
        merged.__dict__.pop(attr, None)
        merged.config.__dict__.pop(attr, None)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.save_pretrained(out)
    print(f"  Merged model saved to {out}")
    return out


def _check_lora_weights(model, label=""):
    """Print norms of a few LoRA parameters to verify they are loaded."""
    lora_norms = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.numel() > 0:
            lora_norms.append((name, param.data.norm().item()))
            if len(lora_norms) >= 4:
                break

    if not lora_norms:
        print(f"  {label}WARNING: No LoRA parameters found!")
        return

    total = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    nonzero = sum(1 for n, p in model.named_parameters()
                  if "lora_" in n and p.data.norm().item() > 1e-8)
    print(f"  {label}LoRA params: {total} total, {nonzero} with non-zero norm")
    for name, norm in lora_norms:
        short = name.split(".")[-3:]
        print(f"    {'·'.join(short):40s} norm={norm:.6f}")


def load_adapter_for_inference(base_model_id: str, adapter_path: str):
    """Reload the quantised base model + LoRA adapter for inference.

    Uses get_peft_model() + manual weight loading instead of
    PeftModel.from_pretrained() to work around a PEFT bug where the resolved
    target_modules list in adapter_config.json fails to inject LoRA wrappers
    into Gemma4ForConditionalGeneration.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=compute_dtype,
    )

    print(f"  Loading quantised base model: {base_model_id}")
    base = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=compute_dtype,
    )

    # Build LoRA config from saved adapter, using dynamic target resolution
    cfg = _adapter_config_override(adapter_path)
    print(f"  Injecting LoRA via get_peft_model (target_modules='all-linear')...")
    model = get_peft_model(base, cfg)

    # Load saved adapter weights
    weights_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(weights_file):
        weights_file = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(weights_file, map_location="cpu", weights_only=True)
    else:
        adapter_weights = load_safetensors(weights_file)

    set_peft_model_state_dict(model, adapter_weights)
    model.eval()
    _check_lora_weights(model, label="After manual load: ")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Quick adapter ON/OFF sanity check
    test_ids = torch.tensor(
        [[2, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107]],
        device=next(model.parameters()).device,
    )
    with torch.no_grad():
        out = model(
            input_ids=test_ids,
            attention_mask=torch.ones_like(test_ids),
            token_type_ids=torch.zeros_like(test_ids),
            mm_token_type_ids=torch.zeros_like(test_ids),
        )
        on_id = out.logits[0, -1].argmax().item()

        with model.disable_adapter():
            out_off = model(
                input_ids=test_ids,
                attention_mask=torch.ones_like(test_ids),
                token_type_ids=torch.zeros_like(test_ids),
                mm_token_type_ids=torch.zeros_like(test_ids),
            )
            off_id = out_off.logits[0, -1].argmax().item()

    print(f"  Adapter ON:  {tokenizer.decode([on_id])!r}  (id={on_id})")
    print(f"  Adapter OFF: {tokenizer.decode([off_id])!r}  (id={off_id})")
    if on_id == off_id:
        print("  WARNING: adapter ON/OFF identical — LoRA not affecting forward!")
        for name, mod in model.named_modules():
            if "q_proj" in name and "language_model" in name and "layers.0" in name:
                print(f"    {name}: {type(mod).__module__}.{type(mod).__qualname__}")
                print(f"    has lora_A: {hasattr(mod, 'lora_A')}")
                break

    return model, tokenizer


def run_inference_test(
    model_path: str,
    dataset,
    n_samples: int = 3,
    base_model_id: str | None = None,
):
    """Run sample inferences.

    If *base_model_id* is provided the adapter at *model_path* is loaded on
    top of the quantised base model (recommended for QLoRA).  Otherwise
    *model_path* is treated as a standalone merged model.
    """
    print(f"\nRunning inference test on {n_samples} samples ...")

    if base_model_id is not None:
        model, tokenizer = load_adapter_for_inference(base_model_id, model_path)
    else:
        print(f"  Loading merged model: {model_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, device_map="auto", dtype="auto"
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    _generate_samples(
        model, tokenizer, dataset,
        n_samples=n_samples, max_new_tokens=64, show_tokens=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune Gemma with QLoRA (HF Transformers + TRL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("model")
    g.add_argument(
        "--model", default="google/gemma-4-E2B",
        help="HuggingFace model ID for the base model",
    )
    g.add_argument(
        "--tokenizer", default=None,
        help="HuggingFace tokenizer ID (defaults to model-id + '-it' for Gemma)",
    )

    g = p.add_argument_group("dataset")
    g.add_argument(
        "--dataset", default="data/data.jsonl",
        help="Path to a local JSONL file or a HuggingFace dataset ID",
    )
    g.add_argument("--max-samples", type=int, default=0,
                   help="Cap on training examples (0 = use all)")
    g.add_argument("--test-size", type=float, default=0.2)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=16)
    g.add_argument("--lora-dropout", type=float, default=0.05)
    g.add_argument("--lora-target-modules", default="all-linear")

    g = p.add_argument_group("training")
    g.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index to train on (single-GPU QLoRA)")
    g.add_argument("--output", default="train/output/gemma-finetune")
    g.add_argument("--epochs", type=int, default=1)
    g.add_argument("--batch-size", type=int, default=4)
    g.add_argument("--gradient-accumulation-steps", type=int, default=1)
    g.add_argument("--learning-rate", type=float, default=5e-5)
    g.add_argument("--max-length", type=int, default=512)
    g.add_argument("--max-grad-norm", type=float, default=0.3)
    g.add_argument("--lr-scheduler", default="constant")
    g.add_argument("--logging-steps", type=int, default=10)
    g.add_argument("--save-strategy", default="epoch")
    g.add_argument("--eval-strategy", default="epoch")
    g.add_argument("--push-to-hub", action="store_true")

    g = p.add_argument_group("post-training")
    g.add_argument(
        "--merge", action="store_true",
        help="Merge LoRA adapter into the base model after training",
    )
    g.add_argument(
        "--merge-output", default=None,
        help="Directory for the merged model (defaults to <output>-merged)",
    )
    g.add_argument(
        "--test-inference", action="store_true",
        help="Run a few inference samples after training (requires --merge)",
    )
    g.add_argument("--test-samples", type=int, default=3)
    g.add_argument(
        "--sample-every", type=int, default=50,
        help="Print model predictions every N training steps (0 = disable)",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None):
    args = parse_args(argv)

    # QLoRA fits on a single GPU — restrict visibility so the Trainer
    # doesn't wrap the model in DataParallel across multiple devices.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # ---- 1. Dataset -------------------------------------------------------
    splits = prepare_dataset(
        args.dataset, args.max_samples, args.test_size, args.seed
    )

    # ---- 2. Model + tokenizer ---------------------------------------------
    model, tokenizer, compute_dtype = load_model_and_tokenizer(
        args.model, args.tokenizer
    )

    # ---- 3. LoRA config ---------------------------------------------------
    peft_config = build_lora_config(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    # ---- 4. Training args -------------------------------------------------
    training_args = build_training_args(
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        compute_dtype=compute_dtype,
        push_to_hub=args.push_to_hub,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler,
    )

    # ---- 5. Train ---------------------------------------------------------
    callbacks = []
    if args.sample_every > 0:
        callbacks.append(
            SampleGenerationCallback(
                tokenizer, splits["test"],
                every=args.sample_every, n_samples=2,
            )
        )

    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Show PEFT info
    print(f"\nModel type : {type(trainer.model).__name__}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    if hasattr(trainer.model, "active_adapters"):
        print(f"Active adapters: {trainer.model.active_adapters}")

    # Pre-training baseline (before any weight updates)
    print("\n--- PRE-TRAINING baseline (LoRA weights at init) ---")
    _generate_samples(
        trainer.model, tokenizer, splits["test"],
        n_samples=1, label="[baseline] ", show_tokens=True,
    )

    trainer.train()
    trainer.save_model()
    print(f"\nAdapter saved to {args.output}")

    # ---- 6. In-memory smoke test (no save/load) ---------------------------
    if args.test_inference:
        _check_lora_weights(trainer.model, label="In-memory after training: ")

        print("\n" + "=" * 60)
        print("  IN-MEMORY test (trainer model, no save/load cycle)")
        print("=" * 60)
        _generate_samples(
            trainer.model, tokenizer, splits["test"],
            n_samples=args.test_samples, max_new_tokens=64, show_tokens=True,
        )

    # ---- 7. Cleanup -------------------------------------------------------
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 8. Optional merge ------------------------------------------------
    merge_dir = None
    if args.merge:
        merge_dir = args.merge_output or f"{args.output}-merged"
        merge_adapter(args.model, args.output, merge_dir)

    # ---- 9. Optional inference test (from disk) ---------------------------
    if args.test_inference:
        print("\n" + "=" * 60)
        print("  ADAPTER test (fresh load from disk)")
        print("=" * 60)
        run_inference_test(
            args.output,
            splits["test"],
            args.test_samples,
            base_model_id=args.model,
        )

        if merge_dir is not None:
            print("\n" + "=" * 60)
            print("  MERGED model test (fresh load from disk)")
            print("=" * 60)
            run_inference_test(
                str(merge_dir), splits["test"], args.test_samples
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
