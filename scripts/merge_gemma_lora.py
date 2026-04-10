"""
Merge a LoRA adapter into a Gemma 4 base model and save the merged weights.

Steps:
  1. Load the full multimodal base model.
  2. Unwrap Gemma4ClippableLinear → nn.Linear so PEFT can inject LoRA.
  3. Load adapter via PeftModel.from_pretrained() and merge_and_unload().
  4. Save merged model, then remap keys to restore .linear.weight format
     that vLLM and from_pretrained expect.

See https://github.com/huggingface/peft/issues/3129

Usage (standalone):
    python scripts/merge_gemma_lora.py \\
        --adapter-path train/output/gemma-finetune \\
        --output-path train/output/gemma-merged \\
        --base-model google/gemma-4-E4B-it

Also called from scripts/merge_lora.py -> scripts/convert_to_ollama.sh.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor


def _unwrap_clippable_linear(model) -> int:
    """Replace Gemma4ClippableLinear with inner nn.Linear for PEFT compat."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, Gemma4ClippableLinear):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], module.linear)
            count += 1
    return count


def _load_base_safetensors(base_path: str) -> dict:
    """Load all safetensors from a model directory into a single dict."""
    single = os.path.join(base_path, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    weights = {}
    for f in sorted(Path(base_path).glob("model-*.safetensors")):
        weights.update(load_file(str(f)))
    return weights


def _resolve_base_local_path(base_model_id: str) -> str:
    """Return a local directory for the base model weights.

    If base_model_id is already a local path, return it.  Otherwise
    use huggingface_hub to ensure the model is cached locally.
    """
    if os.path.isdir(base_model_id):
        return base_model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(base_model_id)


def merge_gemma_lora(adapter_path: Path, output_path: Path, base_model_id: str):
    """Load Gemma 4 base + LoRA adapter, merge, and save."""
    adapter_dir = str(adapter_path)
    merged_dir = str(output_path)
    os.makedirs(merged_dir, exist_ok=True)

    print(f"Base model: {base_model_id}")
    print(f"Adapter dir: {adapter_dir}")
    print(f"Merged dir: {merged_dir}")

    # Step 1: Load base model
    print("Loading base model...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"  Model class: {type(model).__name__}")

    # Step 2: Unwrap Gemma4ClippableLinear → nn.Linear for PEFT
    count = _unwrap_clippable_linear(model)
    print(f"Unwrapped {count} Gemma4ClippableLinear modules for PEFT.")

    # Step 3: Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("LoRA merged successfully.")

    # Step 4: Save merged model
    model.save_pretrained(merged_dir, safe_serialization=True)

    # Step 5: Remap keys to match what vLLM / from_pretrained expect.
    # Use the base model's checkpoint keys as ground truth.
    base_local = _resolve_base_local_path(base_model_id)
    base_weights = _load_base_safetensors(base_local)
    base_keys = set(base_weights.keys())

    merged_path = os.path.join(merged_dir, "model.safetensors")
    if os.path.exists(merged_path):
        weights = load_file(merged_path)
    else:
        weights = {}
        for f in sorted(Path(merged_dir).glob("model-*.safetensors")):
            weights.update(load_file(str(f)))

    remapped = {}
    remap_count = 0
    for key, tensor in weights.items():
        if key in base_keys:
            remapped[key] = tensor
        else:
            new_key = (
                key.rsplit(".weight", 1)[0] + ".linear.weight"
                if key.endswith(".weight") else key
            )
            if new_key in base_keys:
                remapped[new_key] = tensor
                remap_count += 1
            else:
                remapped[key] = tensor

    # Fill any missing keys from the base model.  Gemma 4 uses weight tying
    # (shared K/V projections across layer groups), so save_pretrained only
    # writes each unique tensor once.  The original checkpoint has explicit
    # copies for every layer — we need those for GGUF conversion / vLLM.
    fill_count = 0
    for key in base_keys:
        if key not in remapped:
            remapped[key] = base_weights[key]
            fill_count += 1
    if fill_count:
        print(f"Filled {fill_count} missing keys from base model (tied/shared weights).")

    print(f"Remapped {remap_count} keys to match vLLM format.")
    save_file(remapped, merged_path)

    # Clean up shard files if we consolidated
    for f in Path(merged_dir).glob("model-*.safetensors"):
        f.unlink()
    index = os.path.join(merged_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        os.remove(index)

    print(f"Saved merged model to {merged_dir}")

    # Verify
    merged_weights = load_file(merged_path)
    for key in sorted(merged_weights.keys()):
        if "language_model.layers.0.self_attn.q_proj" in key:
            base_norm = base_weights[key].float().norm().item()
            merged_norm = merged_weights[key].float().norm().item()
            diff = (merged_weights[key].float() - base_weights[key].float()).norm().item()
            print(f"Verify: {key}")
            print(f"  base_norm={base_norm:.4f} merged_norm={merged_norm:.4f} diff_norm={diff:.4f}")
            break

    # Copy processor/tokenizer
    proc_src = adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_local
    try:
        AutoProcessor.from_pretrained(proc_src).save_pretrained(merged_dir)
        print(f"Saved processor from {proc_src}")
    except Exception:
        print(f"Failed to load processor from {proc_src}, falling back to base model")
        AutoProcessor.from_pretrained(base_local).save_pretrained(merged_dir)

    for fn in ("config.json", "generation_config.json"):
        src = os.path.join(adapter_dir, fn)
        if os.path.exists(src):
            print(f"Copying {src} to {os.path.join(merged_dir, fn)}")
            shutil.copy2(src, os.path.join(merged_dir, fn))

    print("Done! Merged Gemma 4 model saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into a Gemma 4 base model",
    )
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace model ID (auto-detected from adapter_config.json if omitted)",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if args.base_model:
        base_model_id = args.base_model
    else:
        with open(adapter_path / "adapter_config.json") as f:
            cfg = json.load(f)
        base_model_id = cfg["base_model_name_or_path"]
        print(f"Auto-detected base model: {base_model_id}")

    merge_gemma_lora(adapter_path, Path(args.output_path), base_model_id)


if __name__ == "__main__":
    main()
