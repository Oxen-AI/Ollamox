"""
Merge a LoRA adapter into the base model and save the merged weights.

For Gemma 4 models, automatically delegates to merge_gemma_lora which
remaps adapter keys and uses the text-only Gemma4ForCausalLM for merging.

Usage:
    python scripts/merge_lora.py \
        --adapter-path models/ox-zesty-white-chipmunk \
        --output-path models/ox-zesty-white-chipmunk-merged \
        --base-model Qwen/Qwen3.5-0.8B
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_base_model(adapter_path: Path, explicit_base: str | None) -> str:
    if explicit_base:
        return explicit_base
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    raw = config["base_model_name_or_path"]
    # The training script saved the path with underscores instead of slashes.
    # Convert "Qwen_Qwen3.5-0.8B_local" -> "Qwen/Qwen3.5-0.8B"
    base_model_id = raw.replace("_local", "").replace("_", "/", 1)
    print(f"Inferred base model from adapter config: {base_model_id}")
    return base_model_id


def _is_gemma4(base_model_id: str) -> bool:
    return "gemma-4" in base_model_id.lower()


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="models/ox-zesty-white-chipmunk",
        help="Path to the LoRA adapter directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="models/ox-zesty-white-chipmunk-merged",
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
    base_model_id = _resolve_base_model(adapter_path, args.base_model)

    if _is_gemma4(base_model_id):
        from merge_gemma_lora import merge_gemma_lora

        print(f"Detected Gemma 4 model — using Gemma-specific merge path")
        merge_gemma_lora(adapter_path, output_path, base_model_id)
        return

    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)
    original_base = config["base_model_name_or_path"]
    if original_base != base_model_id:
        config["base_model_name_or_path"] = base_model_id
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Updated adapter_config.json base_model_name_or_path: {original_base} -> {base_model_id}")

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Always load the tokenizer from the base model. The adapter's
    # tokenizer_config.json may have an invalid tokenizer_class
    # (e.g. "TokenizersBackend") that transformers can't instantiate.
    print("Loading tokenizer from the base model")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Copy the chat template if present
    chat_template = adapter_path / "chat_template.jinja"
    if chat_template.exists():
        shutil.copy2(chat_template, output_path / "chat_template.jinja")
        print("Copied chat_template.jinja to merged model directory")

    print("Done! Merged model saved successfully.")


if __name__ == "__main__":
    main()
