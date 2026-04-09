"""
Test a merged HuggingFace model directly — no GGUF, no Ollama.

Loads the merged model, runs a few prompts through it, and prints the
responses. Use this to verify the LoRA merge worked before spending time
on GGUF conversion. Also loads the base model for side-by-side comparison
so you can see exactly what the fine-tune changed.

Usage:
    python scripts/test_merged_model.py \
        --merged-path models/ox-gemma4-merged

    python scripts/test_merged_model.py \
        --merged-path models/ox-gemma4-merged \
        --base-model google/gemma-4-E2B-it \
        --prompt "Hello"
"""

import argparse
import json
from pathlib import Path

import torch


def load_model_and_tokenizer(model_path: str, model_class=None):
    from transformers import AutoTokenizer

    if model_class is None:
        from transformers import AutoModelForImageTextToText
        model_class = AutoModelForImageTextToText

    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def compare_weights(merged_path: str, base_path: str):
    """Load both models and compare a few weight tensors."""
    from safetensors import safe_open

    merged_dir = Path(merged_path)
    merged_files = sorted(merged_dir.glob("*.safetensors"))
    if not merged_files:
        print("  No safetensors files in merged model directory")
        return

    # Collect merged weights for language model layers
    merged_weights = {}
    for sf in merged_files:
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if "self_attn.q_proj.weight" in key and "vision" not in key and "audio" not in key:
                    merged_weights[key] = f.get_tensor(key)

    # Try to load matching base weights from HF cache
    try:
        from transformers import AutoModelForImageTextToText
        print(f"  Loading base model for weight comparison: {base_path}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base_state = base_model.state_dict()

        changed = 0
        unchanged = 0
        for key, merged_tensor in sorted(merged_weights.items())[:10]:
            if key in base_state:
                delta = (merged_tensor.float() - base_state[key].float())
                delta_norm = delta.norm().item()
                base_norm = base_state[key].float().norm().item()
                pct = (delta_norm / base_norm * 100) if base_norm > 0 else 0
                status = "CHANGED" if delta_norm > 1e-8 else "UNCHANGED"
                if delta_norm > 1e-8:
                    changed += 1
                else:
                    unchanged += 1
                print(f"    {key}")
                print(f"      base norm: {base_norm:.4f}  delta norm: {delta_norm:.6f}  ({pct:.3f}%)  [{status}]")
            else:
                print(f"    {key} — not found in base model")

        remaining = len(merged_weights) - 10
        if remaining > 0:
            print(f"    ... ({remaining} more q_proj layers)")
        print(f"  Summary: {changed} changed, {unchanged} unchanged out of first 10 q_proj layers")
        del base_model, base_state
    except Exception as e:
        print(f"  Could not load base model for comparison: {e}")
        print("  Showing merged weight stats only:")
        for key, tensor in sorted(merged_weights.items())[:5]:
            print(f"    {key}  norm={tensor.float().norm().item():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a merged HuggingFace model directly (no GGUF)"
    )
    parser.add_argument(
        "--merged-path",
        type=str,
        required=True,
        help="Path to the merged model directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HuggingFace base model ID for side-by-side comparison. "
        "If not provided, reads from the merged model's config.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help="Prompt(s) to test. Can be specified multiple times. "
        "Defaults to a few generic prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip loading/testing the base model (faster, less memory)",
    )
    args = parser.parse_args()

    merged_path = Path(args.merged_path)
    prompts = args.prompt or ["Hello", "What is 2+2?"]

    # Try to find the base model from the merged model's config
    base_model_id = args.base_model
    if base_model_id is None:
        config_path = merged_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            # Gemma 4 config nests the model name in _name_or_path
            base_model_id = config.get("_name_or_path", "")
            if base_model_id:
                print(f"Base model from config: {base_model_id}")

    # --- Weight comparison ---
    if base_model_id and not args.skip_base:
        print("\n=== Weight Comparison (merged vs base) ===")
        compare_weights(str(merged_path), base_model_id)

    # --- Test merged model ---
    print("\n=== Loading Merged Model ===")
    merged_model, merged_tokenizer = load_model_and_tokenizer(str(merged_path))
    print(f"  Model class: {type(merged_model).__name__}")
    print(f"  Device: {merged_model.device}")

    print("\n=== Merged Model Responses ===")
    for prompt in prompts:
        response = generate(merged_model, merged_tokenizer, prompt, args.max_new_tokens)
        print(f"\n  Prompt: {prompt}")
        print(f"  Response: {response[:500]}")

    # --- Test base model for comparison ---
    if base_model_id and not args.skip_base:
        del merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("\n=== Loading Base Model for Comparison ===")
        base_model, base_tokenizer = load_model_and_tokenizer(base_model_id)
        print(f"  Model class: {type(base_model).__name__}")

        print("\n=== Base Model Responses ===")
        for prompt in prompts:
            response = generate(base_model, base_tokenizer, prompt, args.max_new_tokens)
            print(f"\n  Prompt: {prompt}")
            print(f"  Response: {response[:500]}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
