"""
Test LoRA merge pipeline for Gemma 4 at every stage:

  1. Base model alone (optional, skipped with --skip-base)
  2. Adapter loaded via PeftModel (NOT merged) — forward_generate()
  3. After merge_and_unload() in memory — forward_generate()
  4. Save merged to temp dir → remap keys → reload from disk → test
  5. Pre-existing merged model from --merged-path (optional)

Uses the same unwrap-ClippableLinear + PeftModel.from_pretrained approach
as merge_gemma_lora.py.

Usage:
    python scripts/test_merged_model.py \\
        --adapter-path train/output/gemma-finetune-fp \\
        --prompt "Hi"

    python scripts/test_merged_model.py \\
        --merged-path train/output/gemma-finetune-fp-merged \\
        --prompt "Hi"
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _save_with_key_remap(model, output_dir: str, original_keys: set):
    """Save merged (unwrapped) model, remapping keys to restore
    .linear.weight format so from_pretrained can reload it."""
    model.save_pretrained(output_dir, safe_serialization=True)

    merged_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(merged_path):
        weights = load_file(merged_path)
    else:
        weights = {}
        for f in sorted(Path(output_dir).glob("model-*.safetensors")):
            weights.update(load_file(str(f)))

    remapped = {}
    remap_count = 0
    for key, tensor in weights.items():
        if key in original_keys:
            remapped[key] = tensor
        else:
            new_key = (
                key.rsplit(".weight", 1)[0] + ".linear.weight"
                if key.endswith(".weight") else key
            )
            if new_key in original_keys:
                remapped[new_key] = tensor
                remap_count += 1
            else:
                remapped[key] = tensor

    save_file(remapped, merged_path)
    for f in Path(output_dir).glob("model-*.safetensors"):
        f.unlink()
    index = os.path.join(output_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        os.remove(index)
    print(f"  Remapped {remap_count} keys to match original checkpoint format.")


@torch.no_grad()
def forward_generate(model, tokenizer, input_ids, attention_mask=None,
                     max_new_tokens=128):
    """Greedy decode via model(**inputs) — bypasses the model.generate() bug
    where Gemma4's prepare_inputs_for_generation skips the PEFT wrapper."""
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
            torch.ones((1, 1), device=generated.device, dtype=attention_mask.dtype),
        ], dim=-1)
        if next_id.item() in eos_ids:
            break

    return generated


def run_prompt(model, tokenizer, prompt: str, max_new_tokens: int,
               use_forward_generate: bool = False) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    if use_forward_generate:
        output_ids = forward_generate(
            model, tokenizer, inputs["input_ids"],
            inputs.get("attention_mask"), max_new_tokens,
        )
    else:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def test_stage(label: str, model, tokenizer, prompts: list[str],
               max_new_tokens: int, use_forward_generate: bool = False):
    method = "forward_generate" if use_forward_generate else "model.generate"
    print(f"\n{'='*60}")
    print(f"  {label}  [{method}]")
    print(f"{'='*60}")
    for prompt in prompts:
        response = run_prompt(model, tokenizer, prompt, max_new_tokens,
                              use_forward_generate)
        print(f"\n  Prompt:   {prompt}")
        print(f"  Response: {response[:500]}")


def load_base(base_model_id: str, device_map=None):
    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
    print(f"  Loading: {base_model_id} (bfloat16, device_map={device_map})")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print(f"  Class: {type(model).__name__}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test Gemma 4 LoRA merge at every stage",
    )
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--merged-path", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--prompt", type=str, action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip the bare base-model test")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Only test adapter, don't merge")
    args = parser.parse_args()

    if not args.adapter_path and not args.merged_path:
        parser.error("Provide at least one of --adapter-path or --merged-path")

    prompts = args.prompt or ["Hi"]

    base_model_id = args.base_model
    if base_model_id is None and args.adapter_path:
        with open(Path(args.adapter_path) / "adapter_config.json") as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
        print(f"Base model (from adapter config): {base_model_id}")

    # ==================================================================
    # Stage A: adapter-path tests
    # ==================================================================
    if args.adapter_path:
        adapter_dir = str(args.adapter_path)

        # --- A1: Base model alone (optional) ---
        if not args.skip_base and base_model_id:
            print("\n>>> STAGE 1: Base model (no adapter)")
            base, tokenizer = load_base(base_model_id)
            test_stage("BASE MODEL (no adapter)", base, tokenizer,
                       prompts, args.max_new_tokens,
                       use_forward_generate=True)
            del base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- A2: Adapter loaded (NOT merged) ---
        print("\n>>> STAGE 2: PeftModel (adapter loaded, NOT merged)")
        base, tokenizer = load_base(base_model_id)

        # Capture original keys before unwrapping (for key remap later)
        original_keys = (
            {n for n, _ in base.named_parameters()}
            | {n for n, _ in base.named_buffers()}
        )

        count = _unwrap_clippable_linear(base)
        print(f"  Unwrapped {count} Gemma4ClippableLinear modules for PEFT.")

        print(f"  Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        lora_count = sum(
            1 for _, m in model.named_modules()
            if hasattr(m, "lora_A") and m.lora_A
        )
        print(f"  {lora_count} LoRA modules with loaded weights")

        test_stage("PEFT MODEL (adapter loaded, NOT merged)",
                   model, tokenizer, prompts, args.max_new_tokens,
                   use_forward_generate=True)

        # --- A3: merge_and_unload in memory ---
        if not args.skip_merge:
            print("\n>>> STAGE 3: After merge_and_unload()")
            merged = model.merge_and_unload()
            test_stage("MERGED MODEL (in memory, after merge_and_unload)",
                       merged, tokenizer, prompts, args.max_new_tokens,
                       use_forward_generate=True)

            # --- A4: save merged to disk → remap keys → reload → test ---
            print("\n>>> STAGE 4: Save merged to disk → remap keys → reload → test")
            with tempfile.TemporaryDirectory() as tmp_dir:
                print(f"  Saving merged model to: {tmp_dir}")
                _save_with_key_remap(merged, tmp_dir, original_keys)
                tokenizer.save_pretrained(tmp_dir)
                del merged
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"  Reloading merged model from disk...")
                disk_device = {"": 0} if torch.cuda.is_available() else "cpu"
                reloaded = AutoModelForImageTextToText.from_pretrained(
                    tmp_dir,
                    torch_dtype=torch.bfloat16,
                    device_map=disk_device,
                )
                reloaded_tok = AutoTokenizer.from_pretrained(tmp_dir)
                print(f"  Class: {type(reloaded).__name__}")

                test_stage("MERGED MODEL (saved to disk → reloaded) — forward_generate",
                           reloaded, reloaded_tok, prompts, args.max_new_tokens,
                           use_forward_generate=True)

                test_stage("MERGED MODEL (saved to disk → reloaded) — model.generate",
                           reloaded, reloaded_tok, prompts, args.max_new_tokens,
                           use_forward_generate=False)

                del reloaded

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================================
    # Stage B: test a pre-existing merged model from disk
    # ==================================================================
    if args.merged_path:
        print("\n>>> STAGE 5: Pre-existing merged model from disk")
        print(f"  Loading: {args.merged_path}")
        disk_device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        merged = AutoModelForImageTextToText.from_pretrained(
            args.merged_path,
            torch_dtype=torch.bfloat16,
            device_map=disk_device_map,
        )

        try:
            processor = AutoProcessor.from_pretrained(args.merged_path)
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.merged_path)
        print(f"  Class: {type(merged).__name__}")

        test_stage("PRE-EXISTING MERGED MODEL (from disk) — forward_generate",
                   merged, tokenizer, prompts, args.max_new_tokens,
                   use_forward_generate=True)

        test_stage("PRE-EXISTING MERGED MODEL (from disk) — model.generate",
                   merged, tokenizer, prompts, args.max_new_tokens,
                   use_forward_generate=False)

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
