"""
Merge a LoRA adapter into a Gemma 4 base model and save the merged weights.

Gemma 4 adapters are trained against Gemma4ForConditionalGeneration, which
nests the text decoder under `model.language_model.*`.  Even when loaded via
AutoModelForCausalLM, HuggingFace returns Gemma4ForConditionalGeneration for
multimodal checkpoints like google/gemma-4-E2B-it, so the adapter keys match
without remapping.

The vision/audio encoders use Gemma4ClippableLinear (inherits nn.Module, not
nn.Linear), which PEFT rejects during its type check.  We monkey-patch the
class to inherit from nn.Linear so PEFT treats it as a supported target.

See: https://github.com/huggingface/peft/issues/3129
See: https://github.com/vllm-project/vllm/pull/38844
See: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

Usage:
    python scripts/merge_gemma_lora.py \\
        --adapter-path models/ox-gemma4-adapter \\
        --output-path models/ox-gemma4-merged \\
        --base-model google/gemma-4-E2B-it
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn


def _patch_gemma4_clippable_linear():
    """Monkey-patch Gemma4ClippableLinear to inherit from nn.Linear so PEFT
    recognises it as a supported target module.

    The original class stores weights in a sub-module (self.linear.weight).
    Our patched version IS an nn.Linear (self.weight), but we override
    _load_from_state_dict to remap the original checkpoint keys so that
    vision/audio weights load correctly.
    """
    from transformers.models.gemma4 import modeling_gemma4

    class PatchedClippableLinear(nn.Linear):
        def __init__(self, config, in_features, out_features):
            nn.Linear.__init__(self, in_features, out_features, bias=False)
            self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
            if self.use_clipped_linears:
                self.register_buffer("input_min", torch.tensor(-float("inf")))
                self.register_buffer("input_max", torch.tensor(float("inf")))
                self.register_buffer("output_min", torch.tensor(-float("inf")))
                self.register_buffer("output_max", torch.tensor(float("inf")))

        def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        ):
            for suffix in ("weight", "bias"):
                old_key = f"{prefix}linear.{suffix}"
                new_key = f"{prefix}{suffix}"
                if old_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            )

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear


def _verify_lora_applied(model) -> int:
    """Count LoRA-adapted modules with non-zero B matrices."""
    lora_count = 0
    loaded_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_count += 1
            for key, param in module.lora_B.items():
                if param.weight.any():
                    loaded_count += 1
                    break
    if lora_count == 0:
        print("WARNING: No LoRA-adapted modules found!")
    elif loaded_count == 0:
        print(f"WARNING: Found {lora_count} LoRA modules but ALL have zero B-matrices. "
              "Adapter weights were not loaded.")
    else:
        print(f"Verified: {loaded_count}/{lora_count} LoRA modules have trained weights")
    return loaded_count


def _dump_model_modules(model, pattern="q_proj", limit=5):
    """Print a few representative module paths that match a pattern."""
    matches = [name for name, _ in model.named_modules() if pattern in name.split(".")[-1]]
    print(f"  Model has {len(matches)} modules matching '{pattern}'. First {limit}:")
    for m in matches[:limit]:
        print(f"    {m}")
    if len(matches) > limit:
        print(f"    ... ({len(matches) - limit} more)")


def merge_gemma_lora(adapter_path: Path, output_path: Path, base_model_id: str):
    """Load Gemma 4 base model + LoRA adapter, merge, and save."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_gemma4_clippable_linear()

    # AutoModelForCausalLM returns Gemma4ForConditionalGeneration for
    # multimodal checkpoints.  The adapter keys already use the
    # `model.language_model.*` naming, so no remapping is needed.
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  Model class: {type(base_model).__name__}")
    _dump_model_modules(base_model)

    # Show adapter config
    print(f"\nLoading LoRA adapter from: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    original_base = adapter_cfg["base_model_name_or_path"]
    if original_base != base_model_id:
        adapter_cfg["base_model_name_or_path"] = base_model_id
        with open(config_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)
        print(f"  Updated adapter_config base: {original_base} -> {base_model_id}")

    print(f"  target_modules: {adapter_cfg.get('target_modules')}")
    print(f"  task_type: {adapter_cfg.get('task_type')}")
    print(f"  r: {adapter_cfg.get('r')}, lora_alpha: {adapter_cfg.get('lora_alpha')}")

    # Snapshot a base weight for post-merge diff
    base_snapshot_name = None
    base_snapshot_data = None
    for name, param in base_model.named_parameters():
        if "self_attn.q_proj.weight" in name and "vision" not in name and "audio" not in name:
            base_snapshot_name = name
            base_snapshot_data = param.data.clone()
            break

    # Apply adapter
    print("\nApplying adapter...")
    model = PeftModel.from_pretrained(
        base_model, str(adapter_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    loaded = _verify_lora_applied(model)
    if loaded == 0:
        raise RuntimeError(
            "LoRA weights not loaded. Check adapter keys vs model module paths."
        )

    # Merge
    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Post-merge diff
    if base_snapshot_name and base_snapshot_data is not None:
        for name, param in model.named_parameters():
            if name == base_snapshot_name:
                delta = (param.data.float() - base_snapshot_data.float())
                delta_norm = delta.norm().item()
                base_norm = base_snapshot_data.float().norm().item()
                pct = (delta_norm / base_norm * 100) if base_norm > 0 else 0
                print(f"  Post-merge diff on {name}:")
                print(f"    base_norm={base_norm:.4f}  delta_norm={delta_norm:.6f}  ({pct:.4f}%)")
                if delta_norm < 1e-8:
                    print("    WARNING: Weight unchanged — LoRA merge had no effect!")
                else:
                    print("    OK — merge changed the weights")
                break

    # Smoke test
    print("\nRunning quick inference smoke-test on merged model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        response = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"  Prompt: Hello")
        print(f"  Response: {response[:200]}")
    except Exception as e:
        print(f"  Smoke-test skipped: {e}")

    # Save
    print(f"\nSaving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")

    # Save tokenizer
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        print("  Loaded tokenizer from adapter directory")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        print("  Loaded tokenizer from base model")
    tokenizer.save_pretrained(output_path)

    chat_template = adapter_path / "chat_template.jinja"
    if chat_template.exists():
        shutil.copy2(chat_template, output_path / "chat_template.jinja")
        print("Copied chat_template.jinja to merged model directory")

    print("\nDone! Merged Gemma 4 model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Gemma 4 base model")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace model ID. If omitted, reads from adapter_config.json.",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)

    if args.base_model:
        base_model_id = args.base_model
    else:
        config_path = adapter_path / "adapter_config.json"
        with open(config_path) as f:
            config = json.load(f)
        raw = config["base_model_name_or_path"]
        base_model_id = raw.replace("_local", "").replace("_", "/", 1)
        print(f"Inferred base model from adapter config: {base_model_id}")

    merge_gemma_lora(adapter_path, output_path, base_model_id)


if __name__ == "__main__":
    main()
