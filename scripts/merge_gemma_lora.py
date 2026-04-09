"""
Merge a LoRA adapter into a Gemma 4 base model and save the merged weights.

Gemma 4 is a multimodal model (Gemma4ForConditionalGeneration) that must be
loaded with AutoModelForImageTextToText — NOT AutoModelForCausalLM. The LoRA
adapter is trained against the full multimodal model, so the adapter weight keys
include a `language_model.` prefix. Using the wrong Auto class loads a text-only
model where those keys don't match, and the LoRA silently fails to apply.

Additionally, Gemma 4 introduces Gemma4ClippableLinear (vision/audio encoders)
which inherits from nn.Module instead of nn.Linear. PEFT rejects this during
the type check before exclude_modules filtering runs. We monkey-patch the class
to inherit from nn.Linear so PEFT treats it as a standard linear layer, while
remapping checkpoint keys in _load_from_state_dict so vision/audio weights
still load correctly from the original `.linear.weight` key format.

See: https://github.com/huggingface/peft/issues/3129
See: https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

Usage:
    python scripts/merge_gemma_lora.py \
        --adapter-path models/ox-gemma4-adapter \
        --output-path models/ox-gemma4-merged \
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
    vision/audio weights load correctly instead of being randomly initialised.
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
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
            # The original Gemma4ClippableLinear stores its weight in a
            # sub-module: self.linear.weight  (key = prefix + "linear.weight").
            # Our patched version is an nn.Linear, so weight lives directly at
            # prefix + "weight". Remap so the original checkpoint loads cleanly.
            for suffix in ("weight", "bias"):
                old_key = f"{prefix}linear.{suffix}"
                new_key = f"{prefix}{suffix}"
                if old_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
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
    """Count LoRA-adapted modules, and verify the weights are non-trivial
    (i.e. not still at initialisation where B is all zeros)."""
    lora_count = 0
    loaded_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            lora_count += 1
            # lora_B is initialised to zeros. If it has non-zero values,
            # the adapter weights were actually loaded from disk.
            for key, param in module.lora_B.items():
                if param.weight.any():
                    loaded_count += 1
                    break
    if lora_count == 0:
        print(
            "WARNING: No LoRA-adapted modules found! The adapter weights likely "
            "failed to load due to a module-path mismatch."
        )
    elif loaded_count == 0:
        print(
            f"WARNING: Found {lora_count} LoRA modules but ALL have zero B-matrices. "
            "The adapter weights were not loaded — module-path keys likely don't match. "
            "The merged model will be identical to the base model."
        )
    else:
        print(f"Verified: {loaded_count}/{lora_count} LoRA modules have trained weights")
    return loaded_count


def _dump_adapter_keys(adapter_path: Path):
    """Print the first few weight keys from the adapter safetensors file."""
    safetensors_file = adapter_path / "adapter_model.safetensors"
    if not safetensors_file.exists():
        print(f"  (no adapter_model.safetensors found)")
        return
    try:
        from safetensors import safe_open
        with safe_open(str(safetensors_file), framework="pt") as f:
            keys = sorted(f.keys())
        print(f"  Adapter weight file has {len(keys)} keys. First 5:")
        for k in keys[:5]:
            print(f"    {k}")
        if len(keys) > 5:
            print(f"    ... ({len(keys) - 5} more)")
    except ImportError:
        print("  (safetensors package not available, skipping key dump)")


def _dump_model_modules(model, pattern="q_proj", limit=5):
    """Print a few representative module paths that match a pattern."""
    matches = [name for name, _ in model.named_modules() if pattern in name.split(".")[-1]]
    print(f"  Model has {len(matches)} modules matching '{pattern}'. First {limit}:")
    for m in matches[:limit]:
        print(f"    {m}")
    if len(matches) > limit:
        print(f"    ... ({len(matches) - limit} more)")


def merge_gemma_lora(adapter_path: Path, output_path: Path, base_model_id: str):
    """Load a Gemma 4 base model + LoRA adapter, merge, and save."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    _patch_gemma4_clippable_linear()

    # Gemma 4 is a multimodal model (Gemma4ForConditionalGeneration).
    # The LoRA adapter was trained against this full model, so adapter weight
    # keys include the `language_model.` prefix. We MUST load with the same
    # Auto class to get matching module paths.
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  Model class: {type(base_model).__name__}")
    _dump_model_modules(base_model)

    print(f"Loading LoRA adapter from: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        adapter_cfg = json.load(f)
    original_base = adapter_cfg["base_model_name_or_path"]
    if original_base != base_model_id:
        adapter_cfg["base_model_name_or_path"] = base_model_id
        with open(config_path, "w") as f:
            json.dump(adapter_cfg, f, indent=2)
        print(f"  Updated adapter_config.json base_model_name_or_path: {original_base} -> {base_model_id}")

    print(f"  target_modules: {adapter_cfg.get('target_modules')}")
    print(f"  task_type: {adapter_cfg.get('task_type')}")
    print(f"  r: {adapter_cfg.get('r')}, lora_alpha: {adapter_cfg.get('lora_alpha')}")
    _dump_adapter_keys(adapter_path)

    # Snapshot one base weight before LoRA is applied, so we can diff after merge
    base_snapshot_name = None
    base_snapshot_data = None
    for name, param in base_model.named_parameters():
        if "self_attn.q_proj.weight" in name and "vision" not in name and "audio" not in name:
            base_snapshot_name = name
            base_snapshot_data = param.data.clone()
            break

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    loaded = _verify_lora_applied(model)
    if loaded == 0:
        print(
            "\nAdapter weight keys don't match the model's module paths. "
            "This usually means the adapter was trained with a different model "
            "loader class (AutoModelForCausalLM vs AutoModelForImageTextToText). "
            "Retrying with AutoModelForCausalLM..."
        )
        # Clean up and retry with text-only model
        del model, base_model
        from transformers import AutoModelForCausalLM

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  Model class: {type(base_model).__name__}")
        _dump_model_modules(base_model)

        # Re-snapshot base weight for the fallback model
        base_snapshot_name = None
        base_snapshot_data = None
        for name, param in base_model.named_parameters():
            if "self_attn.q_proj.weight" in name and "vision" not in name and "audio" not in name:
                base_snapshot_name = name
                base_snapshot_data = param.data.clone()
                break

        # Re-read adapter config (we may have overwritten it above)
        with open(config_path) as f:
            adapter_cfg = json.load(f)
        if adapter_cfg["base_model_name_or_path"] != base_model_id:
            adapter_cfg["base_model_name_or_path"] = base_model_id
            with open(config_path, "w") as f:
                json.dump(adapter_cfg, f, indent=2)

        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        loaded = _verify_lora_applied(model)
        if loaded == 0:
            print(
                "ERROR: LoRA weights still not loaded after fallback. "
                "The adapter may be corrupt or incompatible."
            )

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Verify the merge actually changed weights
    if base_snapshot_name and base_snapshot_data is not None:
        for name, param in model.named_parameters():
            if name == base_snapshot_name:
                delta = (param.data - base_snapshot_data).float()
                delta_norm = delta.norm().item()
                print(f"  Post-merge diff on {name}:")
                print(f"    delta norm: {delta_norm:.6f}")
                if delta_norm < 1e-8:
                    print("    WARNING: Weight unchanged — LoRA merge had no effect!")
                else:
                    print("    OK — merge changed the weights")
                break

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")

    # Try the adapter's tokenizer first (it may have been updated during
    # training, e.g. new special tokens). Fall back to base model if it fails.
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
        print("  Loaded tokenizer from adapter directory")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        print("  Loaded tokenizer from base model (adapter tokenizer not usable)")
    tokenizer.save_pretrained(output_path)

    chat_template = adapter_path / "chat_template.jinja"
    if chat_template.exists():
        shutil.copy2(chat_template, output_path / "chat_template.jinja")
        print("Copied chat_template.jinja to merged model directory")

    print("Done! Merged Gemma 4 model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Gemma 4 base model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HuggingFace model ID or local path for the base model. "
        "If not provided, reads from adapter_config.json.",
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
