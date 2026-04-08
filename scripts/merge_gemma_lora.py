"""
Merge a LoRA adapter into a Gemma 4 base model and save the merged weights.

Gemma 4 introduces Gemma4ClippableLinear which inherits from nn.Module instead
of nn.Linear. PEFT rejects this during the type check in _create_new_module
*before* exclude_modules filtering runs. The workaround is to monkey-patch the
class to inherit from nn.Linear so PEFT treats it as a standard linear layer.

See: https://github.com/huggingface/peft/issues/3129

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
    recognises it as a supported target module."""
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

        def forward(self, x):
            if self.use_clipped_linears:
                x = torch.clamp(x, self.input_min, self.input_max)
            out = nn.Linear.forward(self, x)
            if self.use_clipped_linears:
                out = torch.clamp(out, self.output_min, self.output_max)
            return out

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear


def merge_gemma_lora(adapter_path: Path, output_path: Path, base_model_id: str):
    """Load a Gemma 4 base model + LoRA adapter, merge, and save."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _patch_gemma4_clippable_linear()

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

    print("Loading tokenizer from the base model")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
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
