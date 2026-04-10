# Gemma 4 QLoRA Fine-Tuning — Known Issues & Fixes

Three bugs prevent QLoRA-trained adapters from working correctly at inference
time with `Gemma4ForConditionalGeneration` (tested with PEFT 0.18.1,
transformers ≥ 4.45, bitsandbytes ≥ 0.43).

---

## 1. `model.generate()` bypasses the PEFT adapter

**Symptom:** Training loss reaches zero and in-training forward passes predict
correctly, but `model.generate()` returns base-model output ("Hi! How can I
help you today?").

**Root cause:** `Gemma4ForConditionalGeneration.generate()` calls its own
`prepare_inputs_for_generation → forward` pipeline, which resolves to the
*unwrapped* base model's `forward()` rather than `PeftModelForCausalLM.forward()`.
The LoRA adapter is never applied during autoregressive generation.

Additionally, Gemma 4's forward pass requires `token_type_ids` and
`mm_token_type_ids` even for text-only inputs. `model.generate()` does not
supply these.

**Fix:** Replace `model.generate()` with a manual greedy-decoding loop that
calls `model(input_ids=..., attention_mask=..., token_type_ids=...,
mm_token_type_ids=...)` at each step — the same code path used during training.
See `forward_generate()` in `train/train.py`.

---

## 2. `PeftModel.from_pretrained()` creates orphan LoRA parameters

**Symptom:** After loading an adapter from disk the LoRA parameter norms look
correct, but toggling the adapter on/off produces *identical* output. The
language-model's `q_proj` is still a bare `bnb.nn.Linear4bit` instead of
PEFT's `peft.tuners.lora.bnb.Linear4bit` wrapper.

**Root cause:** When PEFT resolves `target_modules="all-linear"` during
training, it saves the *resolved* list of ~335 module names into
`adapter_config.json`. For Gemma 4 this list contains a mix of path formats:

```
"language_model.layers.0.self_attn.q_proj"   ← layers 0-15
"32.mlp.gate_proj"                           ← layers 16-41
"linear"                                     ← audio/vision sub-modules
```

When `PeftModel.from_pretrained()` replays this list, its `endswith`-based
matching fails to inject LoRA wrappers into the correct modules. The LoRA
parameters are created but sit as orphan tensors disconnected from the forward
graph.

**Fix:** Bypass `PeftModel.from_pretrained()`. Instead:

1. Load the base model.
2. Call `get_peft_model(base, LoraConfig(target_modules="all-linear", ...))`
   — the same call SFTTrainer makes during training — so PEFT dynamically
   discovers and wraps all linear modules.
3. Load the saved adapter weights with `set_peft_model_state_dict()`, which
   handles the key-format remapping (`.lora_A.weight` → `.lora_A.default.weight`).

Using raw `model.load_state_dict(adapter_weights, strict=False)` does **not**
work because the saved safetensors keys omit the adapter name (`default`).

```python
from peft import get_peft_model, set_peft_model_state_dict

model = get_peft_model(base, LoraConfig(target_modules="all-linear", ...))
adapter_weights = load_safetensors("adapter_model.safetensors")
set_peft_model_state_dict(model, adapter_weights)
```

---

## 3. `merge_and_unload()` re-quantises merged weights (lossy)

**Symptom:** The merged model produces base-model output despite adapter-based
inference working correctly.

**Root cause (two layers):**

1. **Quantisation mismatch:** The LoRA adapter was optimised on **4-bit NF4
   quantised** base weights (`dequant(W_4bit)`). Merging into the
   **full-precision** base (`W_fp`) introduces an error
   `W_fp − dequant(W_4bit)` at every weight.

2. **Re-quantisation in `merge_and_unload()`:** Even after switching to the
   quantised base, PEFT's merge path for `Linear4bit` does:
   `dequant → add LoRA → re-quantise to 4-bit`. The round-trip destroys the
   precise LoRA adjustments before `dequantize()` can recover them.

**Fix:** Load the base with the same 4-bit quantisation, then **manually
merge** each LoRA layer — dequantise the base weight, add the LoRA delta in
float, and store the result directly as a regular `nn.Linear`.  This avoids
re-quantisation entirely: the merged weight is *exactly*
`dequant(W_4bit) + LoRA`, matching the training-time forward pass.

```python
import bitsandbytes as bnb

base = AutoModelForImageTextToText.from_pretrained(
    model_id, quantization_config=bnb_config, device_map={"": 0},
)
peft_model = get_peft_model(base, lora_config)
set_peft_model_state_dict(peft_model, adapter_weights)

for name, module in list(peft_model.base_model.named_modules()):
    if not hasattr(module, "get_base_layer"):
        continue
    base_layer = module.get_base_layer()
    w = bnb.functional.dequantize_4bit(
        base_layer.weight.data, base_layer.weight.quant_state,
    ).to(torch.bfloat16)
    for adapter_name in module.active_adapters:
        w += module.get_delta_weight(adapter_name).to(w.dtype)
    # replace with nn.Linear(w)  — no Params4bit, no re-quantisation
```

---

## Quick reference

| Stage | What broke | Fix |
|---|---|---|
| Inference (generate) | `model.generate()` bypasses PEFT wrapper | Use manual `forward_generate()` loop |
| Adapter loading | `PeftModel.from_pretrained` orphans LoRA params | Use `get_peft_model` + `set_peft_model_state_dict` |
| Adapter key format | Saved keys omit adapter name (`default`) | Use `set_peft_model_state_dict` (not raw `load_state_dict`) |
| Merge | `merge_and_unload()` re-quantises to 4-bit | Manual merge: dequant + add LoRA, keep as float `nn.Linear` |
