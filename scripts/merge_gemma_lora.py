"""
Merge a LoRA adapter into a Gemma 4 base model and save the merged weights.

Gemma 4 adapters are typically trained against the multimodal wrapper
(Gemma4ForConditionalGeneration), which nests the text decoder under
`model.language_model.*`.  The text-only model (Gemma4ForCausalLM) exposes
the same layers under `model.*`.

Loading the full multimodal model for merging brings in vision/audio towers
with Gemma4ClippableLinear layers that PEFT doesn't support natively, and
requires fragile monkey-patching.  Instead, we follow the approach from
https://github.com/vllm-project/vllm/pull/38844 : load the text-only
Gemma4ForCausalLM and remap the adapter keys, stripping the
`language_model.` prefix so they line up with the text-only module paths.
Vision/audio tower adapter keys are dropped (they carry no trained weights
for text-only fine-tunes).

See: https://github.com/huggingface/peft/issues/3129
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
import tempfile
from pathlib import Path

import torch


def _remap_adapter(adapter_path: Path) -> Path:
    """Create a temp copy of the adapter with keys remapped for Gemma4ForCausalLM.

    Transforms adapter keys trained on Gemma4ForConditionalGeneration
    (model.language_model.*) to work with Gemma4ForCausalLM (model.*).
    Drops vision/audio tower keys that carry no trained weights.

    Returns the path to the temp adapter directory.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = adapter_path / "adapter_model.safetensors"
    if not src.exists():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_path}")

    with safe_open(str(src), framework="pt") as f:
        original_keys = sorted(f.keys())
        original_weights = {k: f.get_tensor(k) for k in f.keys()}

    remapped = {}
    dropped = 0
    for key, tensor in original_weights.items():
        if "audio_tower" in key or "vision_tower" in key:
            dropped += 1
            continue
        new_key = key.replace(".model.language_model.", ".model.")
        remapped[new_key] = tensor

    print(f"  Adapter key remapping: {len(original_keys)} original -> "
          f"{len(remapped)} remapped, {dropped} vision/audio keys dropped")
    if remapped:
        sample_old = [k for k in original_keys if "language_model" in k][:2]
        sample_new = sorted(remapped.keys())[:2]
        for old, new in zip(sample_old, sample_new):
            print(f"    {old}")
            print(f"    -> {new}")

    tmpdir = tempfile.mkdtemp(prefix="gemma4_adapter_remapped_")
    save_file(remapped, str(Path(tmpdir) / "adapter_model.safetensors"))

    # Copy and patch adapter_config.json
    with open(adapter_path / "adapter_config.json") as f:
        cfg = json.load(f)
    with open(Path(tmpdir) / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy any other adapter files (tokenizer, etc.)
    for extra in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "tokenizer.model", "chat_template.jinja"):
        src_file = adapter_path / extra
        if src_file.exists():
            shutil.copy2(src_file, Path(tmpdir) / extra)

    return Path(tmpdir)


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
        print("WARNING: No LoRA-adapted modules found! Keys likely don't match.")
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
    """Load Gemma 4 as text-only CausalLM, remap adapter keys, merge, and save."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the text-only model — avoids vision/audio towers and ClippableLinear
    print(f"Loading base model (text-only): {base_model_id}")
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
    with open(adapter_path / "adapter_config.json") as f:
        adapter_cfg = json.load(f)
    print(f"  target_modules: {adapter_cfg.get('target_modules')}")
    print(f"  task_type: {adapter_cfg.get('task_type')}")
    print(f"  r: {adapter_cfg.get('r')}, lora_alpha: {adapter_cfg.get('lora_alpha')}")

    # Remap adapter keys: model.language_model.* -> model.*
    print("\nRemapping adapter keys for text-only model...")
    remapped_path = _remap_adapter(adapter_path)

    # Update the remapped adapter config to point at the correct base model
    remapped_cfg_path = remapped_path / "adapter_config.json"
    with open(remapped_cfg_path) as f:
        remapped_cfg = json.load(f)
    remapped_cfg["base_model_name_or_path"] = base_model_id
    with open(remapped_cfg_path, "w") as f:
        json.dump(remapped_cfg, f, indent=2)

    # Snapshot one base weight before LoRA is applied
    base_snapshot_name = None
    base_snapshot_data = None
    for name, param in base_model.named_parameters():
        if "self_attn.q_proj.weight" in name:
            base_snapshot_name = name
            base_snapshot_data = param.data.clone()
            break

    # Load adapter onto text-only model
    print("\nApplying adapter to text-only model...")
    model = PeftModel.from_pretrained(
        base_model,
        str(remapped_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    loaded = _verify_lora_applied(model)
    if loaded == 0:
        shutil.rmtree(remapped_path, ignore_errors=True)
        raise RuntimeError(
            "LoRA weights not loaded after key remapping. "
            "The adapter may be incompatible with this base model."
        )

    # Merge
    print("\nMerging LoRA weights into base model...")
    model = model.merge_and_unload()

    # Verify the merge changed weights
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
            tokenize=False,
            add_generation_prompt=True,
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

    # Show what was saved
    try:
        from safetensors import safe_open
        saved_files = sorted(output_path.glob("*.safetensors"))
        total_keys = 0
        sample_keys = []
        for sf in saved_files:
            with safe_open(str(sf), framework="pt") as f:
                keys = f.keys()
                total_keys += len(keys)
                if not sample_keys:
                    sample_keys = sorted(keys)[:3]
        print(f"  Saved {total_keys} weight keys across {len(saved_files)} shard(s)")
        print(f"  Sample saved keys: {sample_keys}")
    except Exception:
        pass

    # Save tokenizer
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

    # Cleanup temp dir
    shutil.rmtree(remapped_path, ignore_errors=True)

    print("\nDone! Merged Gemma 4 model saved successfully.")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into Gemma 4 base model")
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
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
