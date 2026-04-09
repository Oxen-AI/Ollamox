"""
Diagnose LoRA merge issues by testing at every stage of the pipeline:

  1. Base model alone (no adapter)
  2. Base + adapter as PeftModel (adapter applied but NOT merged)
  3. After merge_and_unload() (adapter merged into weights)

This isolates whether the problem is in the adapter itself, the merge
step, or something downstream. Also prints detailed LoRA weight stats.

Usage:
    # Full 3-stage test (needs base model + adapter)
    python scripts/test_merged_model.py \
        --adapter-path models/ox-monetary-lavender-finch \
        --prompt "Hi"

    # Just test an already-merged model
    python scripts/test_merged_model.py \
        --merged-path models/ox-monetary-lavender-finch-merged \
        --prompt "Hi" \
        --skip-base
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


def _patch_gemma4_clippable_linear():
    """Same monkey-patch as merge_gemma_lora.py so we can load the adapter."""
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        return

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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
    return tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def inspect_lora_weights(model):
    """Print detailed stats about every LoRA adapter weight."""
    print("\n--- LoRA Weight Inspection ---")
    total_modules = 0
    trained_modules = 0
    rows = []
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        total_modules += 1
        for adapter_key in module.lora_A:
            A = module.lora_A[adapter_key].weight
            B = module.lora_B[adapter_key].weight
            a_norm = A.float().norm().item()
            b_norm = B.float().norm().item()
            ba_norm = (B.float() @ A.float()).norm().item()
            is_trained = B.any().item()
            if is_trained:
                trained_modules += 1
            rows.append((name, A.shape, B.shape, a_norm, b_norm, ba_norm, is_trained))

    # Print summary table — show all untrained and first/last few trained
    lang_trained = [r for r in rows if r[6] and "language_model" in r[0]]
    lang_untrained = [r for r in rows if not r[6] and "language_model" in r[0]]
    other_trained = [r for r in rows if r[6] and "language_model" not in r[0]]
    other_untrained = [r for r in rows if not r[6] and "language_model" not in r[0]]

    print(f"  Total LoRA modules: {total_modules}")
    print(f"  Trained (non-zero B): {trained_modules}")
    print(f"  Language model — trained: {len(lang_trained)}, untrained: {len(lang_untrained)}")
    print(f"  Vision/audio  — trained: {len(other_trained)}, untrained: {len(other_untrained)}")

    if lang_trained:
        scaling = None
        for name, module in model.named_modules():
            if hasattr(module, "scaling"):
                scaling = module.scaling
                break
        if scaling:
            print(f"  LoRA scaling factor: {scaling}")

    def _print_rows(label, row_list, limit=5):
        if not row_list:
            return
        print(f"\n  {label} (showing {min(limit, len(row_list))}/{len(row_list)}):")
        for name, a_shape, b_shape, a_norm, b_norm, ba_norm, trained in row_list[:limit]:
            short = name.replace("base_model.model.", "")
            tag = "TRAINED" if trained else "ZERO-B"
            print(f"    {short}")
            print(f"      A{list(a_shape)} norm={a_norm:.4f}  B{list(b_shape)} norm={b_norm:.6f}  BA norm={ba_norm:.6f}  [{tag}]")

    _print_rows("Language model — trained", lang_trained, 5)
    _print_rows("Language model — UNTRAINED (zero B)", lang_untrained, 3)
    _print_rows("Vision/audio — trained", other_trained, 3)
    _print_rows("Vision/audio — UNTRAINED (zero B)", other_untrained, 3)

    # Aggregate BA norms for trained language model modules
    if lang_trained:
        ba_norms = [r[5] for r in lang_trained]
        print(f"\n  Language model BA norm stats (trained only):")
        print(f"    min={min(ba_norms):.6f}  max={max(ba_norms):.6f}  "
              f"mean={sum(ba_norms)/len(ba_norms):.6f}  "
              f"median={sorted(ba_norms)[len(ba_norms)//2]:.6f}")


def test_stage(label: str, model, tokenizer, prompts, max_new_tokens):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for prompt in prompts:
        response = generate(model, tokenizer, prompt, max_new_tokens)
        print(f"\n  Prompt:   {prompt}")
        print(f"  Response: {response[:500]}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose LoRA merge at every stage (base / PeftModel / merged)"
    )
    parser.add_argument(
        "--adapter-path", type=str, default=None,
        help="Path to the LoRA adapter directory (runs full 3-stage test)",
    )
    parser.add_argument(
        "--merged-path", type=str, default=None,
        help="Path to an already-merged model (skips adapter stages)",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace base model ID. Auto-detected from adapter/merged config if omitted.",
    )
    parser.add_argument(
        "--prompt", type=str, action="append", default=None,
        help="Prompt(s) to test (repeatable). Defaults to ['Hello'].",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
    )
    parser.add_argument(
        "--skip-base", action="store_true",
        help="Skip the base-model-only test (saves memory/time)",
    )
    args = parser.parse_args()

    if not args.adapter_path and not args.merged_path:
        parser.error("Provide at least one of --adapter-path or --merged-path")

    prompts = args.prompt or ["Hello"]

    # --- Resolve base model ID ---
    base_model_id = args.base_model
    if base_model_id is None and args.adapter_path:
        cfg_path = Path(args.adapter_path) / "adapter_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_model_id = cfg.get("base_model_name_or_path", "")
            print(f"Base model from adapter config: {base_model_id}")
    if base_model_id is None and args.merged_path:
        cfg_path = Path(args.merged_path) / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            base_model_id = cfg.get("_name_or_path", "")
            print(f"Base model from merged config: {base_model_id}")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    # =========================================================
    # Stage A: adapter path provided → full 3-stage diagnosis
    # =========================================================
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        _patch_gemma4_clippable_linear()

        # --- A1: Base model alone ---
        if not args.skip_base and base_model_id:
            print("\n>>> STAGE 1: Base model (no adapter)")
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
            print(f"  Model class: {type(base_model).__name__}")

            test_stage("BASE MODEL (no adapter)", base_model, tokenizer, prompts, args.max_new_tokens)

            # Snapshot a weight for later comparison
            base_snapshot = {}
            for name, param in base_model.named_parameters():
                if "self_attn.q_proj.weight" in name and "vision" not in name and "audio" not in name:
                    base_snapshot[name] = param.data.clone().float()
                    if len(base_snapshot) >= 3:
                        break

            del base_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            base_snapshot = {}

        # --- A2: PeftModel (adapter loaded, NOT merged) ---
        print("\n>>> STAGE 2: PeftModel (adapter loaded, NOT merged)")
        from peft import PeftModel

        peft_base = AutoModelForImageTextToText.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        peft_model = PeftModel.from_pretrained(
            peft_base, str(adapter_path),
            torch_dtype=torch.bfloat16,
        )
        peft_model.eval()
        print(f"  Model class: {type(peft_model).__name__}")
        print(f"  Base class:  {type(peft_model.base_model.model).__name__}")

        inspect_lora_weights(peft_model)

        test_stage("PEFT MODEL (adapter loaded, NOT merged)", peft_model, tokenizer, prompts, args.max_new_tokens)

        # --- A3: Merge and test ---
        print("\n>>> STAGE 3: After merge_and_unload()")
        merged_model = peft_model.merge_and_unload()

        # Compare weights with base snapshot
        if base_snapshot:
            print("\n--- Weight diff (merged vs base) ---")
            for name, param in merged_model.named_parameters():
                if name in base_snapshot:
                    delta = (param.data.float() - base_snapshot[name])
                    delta_norm = delta.norm().item()
                    base_norm = base_snapshot[name].norm().item()
                    pct = (delta_norm / base_norm * 100) if base_norm > 0 else 0
                    print(f"  {name}")
                    print(f"    base_norm={base_norm:.4f}  delta_norm={delta_norm:.6f}  ({pct:.4f}%)")

        test_stage("MERGED MODEL (after merge_and_unload)", merged_model, tokenizer, prompts, args.max_new_tokens)

        del merged_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================
    # Stage B: just test an already-saved merged model
    # =========================================================
    if args.merged_path:
        print("\n>>> Testing saved merged model from disk")
        merged_model = AutoModelForImageTextToText.from_pretrained(
            args.merged_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.merged_path, trust_remote_code=True)
        print(f"  Model class: {type(merged_model).__name__}")

        test_stage("SAVED MERGED MODEL (from disk)", merged_model, tokenizer, prompts, args.max_new_tokens)

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
