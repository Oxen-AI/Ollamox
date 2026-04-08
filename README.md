# From Fine-Tuned LoRA Weights to a Local Ollama Model

Fine-tuning open source models puts the power back in your hands. This repository has code to take fine-tuned LoRA weights and import them into Ollama so you can run a fine-tuned model locally on your machine. Running it locally means your data stays on your machine, there are no API costs, and it works offline. You can optimize for accuracy and privacy with full control over the model weights.

## 🦙 What is Ollama?

Ollama is a nice tool for running models locally, but getting your fine-tuned LoRA weights into a format it understands takes a few steps. This guide walks through the full pipeline: downloading your weights from Oxen.ai, merging them into a base model, converting to GGUF, quantizing, and importing into Ollama. If you don't understand what any of that jargon means, don't worry! We'll walk you through step by step.

In this post we will use use a Qwen3.5 fine-tune as our running example, but the process applies generally to any HuggingFace compatible base model.

## Just Give Me the Command

If you don't care about what's happening under the hood and just want your model running in Ollama, download your weights and run the conversion script. This assumes you have already [fine-tuned a model in Oxen.ai](TODO)

```bash
# Download your LoRA adapter weights from Oxen.ai
# oxen download $USERNAME/$REPO_NAME $PATH_TO_WEIGHTS --revision $COMMIT_OR_BRANCH
oxen download ox/Banking-Agent models/ox-frail-salmon-chameleon --revision 543298f2487808e961b5c08166667462

# Merge, convert, quantize, and import into Ollama
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk
```

If your fine-tune was trained with a custom system prompt or tool definitions, you can bake those into the Ollama model so it behaves the same way at inference time:

```bash
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk \
  --tools-path data/tools.json \
  --system-prompt data/system_prompt.txt
```

If you don't pass these flags, the model will run without a system prompt or tool definitions baked in.

When it's done, run your model with `ollama run ox-zesty-white-chipmunk`.

If you want to understand what each step is actually doing, keep reading.

---

## What You'll Need

Before we start, make sure you have these installed:

**Python packages** (for merging LoRA weights):

```bash
pip install torch transformers peft accelerate
```

**Ollama** (for running the model locally):

```bash
brew install ollama   # macOS
# or grab it from https://ollama.com
```

**Oxen CLI** (for downloading weights):

```bash
# Install instructions at https://docs.oxen.ai
```

You'll also need `cmake` and a C++ compiler for building the llama.cpp quantization tool, but the conversion script handles cloning and building llama.cpp automatically.

## The Pipeline at a Glance

Here's what we're doing, end to end:

```
Oxen.ai LoRA weights
    |
    v
[1] Download adapter weights
    |
    v
[2] Merge LoRA into base model (full-weight HF model)
    |
    v
[3] Convert to GGUF (BF16)
    |
    v
[4] Quantize (Q4_K_M)
    |
    v
[5] Generate Modelfile (system prompt + tools + chat template)
    |
    v
[6] ollama create → ollama run
```

Let's walk through each step.

---

## Step 1: Download Your LoRA Weights from Oxen.ai

After a fine-tuning job finishes on Oxen.ai, your trained LoRA adapter weights live in the output repository. You need to pull them down locally.

Use the `oxen download` command to grab the adapter files:

```bash
oxen download ox/ox-zesty-white-chipmunk model \
  --revision main
```

This downloads the `model` directory from the `ox/ox-zesty-white-chipmunk` repository. Swap out the repo name and path for your own fine-tune.

The general form is:

```bash
oxen download <username>/<repo_name> <path/to/folder> \
  --revision <commit_or_branch_name>
```

The `--revision` flag lets you pin to a specific commit or branch, which is useful if you have multiple training runs or checkpoints you want to compare.

After downloading, you should have a directory that looks something like this:

```
models/ox-zesty-white-chipmunk/
  adapter_config.json        # LoRA hyperparameters + base model reference
  adapter_model.safetensors  # The actual LoRA weights
  tokenizer.json             # Tokenizer vocab
  tokenizer_config.json      # Tokenizer settings
  chat_template.jinja        # Chat format template
  training_args.bin          # Training configuration (for reference)
  training_logs.jsonl        # Training metrics (for reference)
```

The two files that matter most are `adapter_config.json` (which tells us the base model and LoRA configuration) and `adapter_model.safetensors` (the trained weights themselves). The adapter weights are tiny compared to the full model -- that's the whole point of LoRA. A 0.8B parameter model might have adapter weights under 50MB.

---

## Step 2: Merge LoRA into the Base Model

LoRA (Low-Rank Adaptation) works by training small rank-decomposition matrices that get applied on top of a frozen base model. To run inference without the PEFT library, we need to merge these adapter weights back into the base model to produce a single, self-contained set of weights.

The `merge_lora.py` script handles this:

```bash
python scripts/merge_lora.py \
  --adapter-path models/ox-zesty-white-chipmunk \
  --output-path models/ox-zesty-white-chipmunk-merged
```

### What's happening under the hood

Let's walk through the important parts of `scripts/merge_lora.py`.

#### 1. Figuring out the base model

The adapter directory doesn't contain a full model -- just the tiny LoRA delta weights. To merge, we need the original base model. The script auto-detects it from `adapter_config.json`:

```python
config_path = adapter_path / "adapter_config.json"
with open(config_path) as f:
    config = json.load(f)
raw = config["base_model_name_or_path"]
# The training script saved the path with underscores instead of slashes.
# Convert "Qwen_Qwen3.5-0.8B_local" -> "Qwen/Qwen3.5-0.8B"
base_model_id = raw.replace("_local", "").replace("_", "/", 1)
```

The training pipeline on Oxen.ai sometimes saves the base model path with underscores instead of slashes (e.g. `Qwen_Qwen3.5-0.8B_local` instead of `Qwen/Qwen3.5-0.8B`). This little string cleanup handles that. If auto-detection doesn't work for your setup, you can always pass `--base-model Qwen/Qwen3.5-0.8B` explicitly.

#### 2. Loading the base model

```python
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
```

This downloads the base model from HuggingFace (or loads it from your local cache) and puts it in memory. Two things worth noting:

- **`torch_dtype=torch.bfloat16`** -- We load in BF16 precision rather than full FP32. This cuts memory usage in half and BF16 is what the model was trained in anyway, so there's no quality loss.
- **`device_map="cpu"`** -- We keep everything on CPU. The merge is a one-time operation and doesn't need GPU acceleration. This also means you can do this step on a machine without a GPU.

#### 3. Loading the LoRA adapter on top

Before loading the adapter, the script patches `adapter_config.json` so PEFT can find the base model:

```python
if original_base != base_model_id:
    config["base_model_name_or_path"] = base_model_id
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
```

Then it layers the LoRA adapter onto the base model:

```python
model = PeftModel.from_pretrained(
    base_model,
    str(adapter_path),
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
```

At this point, `model` is a wrapped object -- the base weights plus the LoRA matrices sitting alongside them. During a forward pass, PEFT would compute `base_weight + (B @ A) * scaling_factor` for each adapted layer. But we don't want that runtime overhead. We want a single, flat set of weights.

#### 4. The actual merge

This is the most important line in the entire script:

```python
model = model.merge_and_unload()
```

`merge_and_unload()` does two things:

1. **Merge** -- For every layer that has a LoRA adapter (in our case, that's `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` -- all the attention and MLP projections), it computes the final weight: `W_merged = W_base + (B @ A) * (alpha / r)`. Here `alpha=16` and `r=16` (from our adapter config), so the scaling factor is 1.0. The low-rank matrices A and B get multiplied together and added directly to the base weight.
2. **Unload** -- It strips away all the PEFT wrapper code and returns a plain `transformers` model. No more adapter scaffolding, no more runtime LoRA computation. Just a normal model with modified weights.

#### 5. Saving the merged model and tokenizer

```python
model.save_pretrained(output_path)
```

This saves the merged weights in HuggingFace's standard safetensors format.

For the tokenizer, the script deliberately loads it from the base model rather than the adapter directory:

```python
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.save_pretrained(output_path)
```

Why not use the adapter's tokenizer? Because the adapter's `tokenizer_config.json` can sometimes have an invalid `tokenizer_class` (like `"TokenizersBackend"`) that `transformers` can't instantiate. The base model's tokenizer is always reliable, and LoRA fine-tuning doesn't change the tokenizer anyway.

Finally, the chat template gets copied over:

```python
chat_template = adapter_path / "chat_template.jinja"
if chat_template.exists():
    shutil.copy2(chat_template, output_path / "chat_template.jinja")
```

This template defines how messages are formatted with the `<|im_start|>` / `<|im_end|>` tokens. It's important to preserve it because the Modelfile generation step later uses it to set up the correct chat format in Ollama.

After this step, `models/ox-zesty-white-chipmunk-merged/` contains a full HuggingFace model that could be loaded with `transformers` directly -- no `peft` required.

---

## Step 3: Convert to GGUF

Ollama doesn't use HuggingFace's format directly. It uses GGUF (GPT-Generated Unified Format), which is the format used by llama.cpp for efficient CPU and GPU inference. We need to convert our merged HuggingFace model into a GGUF file.

The conversion script handles cloning llama.cpp and running the conversion automatically, but here's what each piece does:

### 3a: Set up llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
```

The script clones llama.cpp into `tools/llama.cpp` (or pulls the latest if it already exists). It also installs the Python dependencies needed for conversion:

```bash
pip install --no-deps gguf
pip install numpy sentencepiece protobuf pyyaml tqdm
```

One gotcha here: the script deliberately installs `gguf` with `--no-deps` and avoids using llama.cpp's `requirements.txt`. Why? Because llama.cpp pins `torch~=2.6.0`, which would downgrade your PyTorch and break compatibility with `peft` and `transformers`. We already have torch installed from step 2, so we just grab the `gguf` package by itself.

### 3b: Build the quantize tool

```bash
cmake -S tools/llama.cpp -B tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/llama.cpp/build --target llama-quantize -j $(sysctl -n hw.ncpu)
```

This compiles the `llama-quantize` binary, which we'll need in the next step. It only builds the quantize target, not all of llama.cpp, so it's fairly quick.

### 3c: Convert to BF16 GGUF

```bash
python tools/llama.cpp/convert_hf_to_gguf.py \
  models/ox-zesty-white-chipmunk-merged \
  --outtype bf16 \
  --outfile models/ox-zesty-white-chipmunk-merged/model-bf16.gguf
```

This converts the HuggingFace safetensors into a single GGUF file at BF16 (Brain Float 16) precision. BF16 preserves the full model quality -- no information is lost at this stage. Think of this as a lossless format conversion.

---

## Step 4: Quantize

The BF16 GGUF file is the same size as the original model. For a 0.8B model that's manageable, but for larger models you'll want to quantize to reduce memory usage and speed up inference.

```bash
tools/llama.cpp/build/bin/llama-quantize \
  models/ox-zesty-white-chipmunk-merged/model-bf16.gguf \
  models/ox-zesty-white-chipmunk-merged/model-Q4_K_M.gguf \
  Q4_K_M
```

The default quantization type is **Q4_K_M**, which is a good balance of quality and size. Here's the rough hierarchy:

| Quant Type | Bits | Quality | Size Reduction |
|------------|------|---------|----------------|
| Q8_0       | 8    | Near-lossless | ~50% |
| Q6_K       | 6    | Very good | ~60% |
| Q5_K_M     | 5    | Good | ~65% |
| **Q4_K_M** | **4** | **Good enough for most uses** | **~70%** |
| Q3_K_M     | 3    | Noticeable degradation | ~75% |
| Q2_K       | 2    | Significant degradation | ~80% |

For tool-calling models where precision matters, you might want to stay at Q5_K_M or higher. For casual testing, Q4_K_M works great. You can pass `--quant Q8_0` to the conversion script to change the quantization type.

---

## Step 5: Generate the Modelfile

Ollama uses a `Modelfile` (think of it like a Dockerfile, but for LLMs) to define how a model should be loaded and configured. The `generate_modelfile.py` script creates one that matches what the model was trained on.

This is the step where things get specific to your fine-tune. The generated Modelfile includes:

### The GGUF source

```
FROM ./model-Q4_K_M.gguf
```

### The chat template

```
TEMPLATE """
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

This is ChatML format, which is what Qwen3.5 uses. Getting this right is critical -- if the template doesn't match what the model saw during training, you'll get garbage output. The model learned to respond to specific token patterns like `<|im_start|>` and `<|im_end|>`, so we need to reproduce them exactly.

### Stop tokens

```
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

These tell Ollama when the model has finished generating. Without them, the model will keep going and start hallucinating the next turn of conversation.

### Generation parameters

```
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 32768
```

These are tuned for the Qwen3.5 models. `num_ctx` sets the context window to 32K tokens.

### The system prompt with tools baked in

This is where it gets interesting. The script reads `data/tools.json` and `data/system_prompt.txt` and bakes the tool definitions directly into the system prompt, formatted exactly as Qwen3.5 expects:

```
SYSTEM """
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_balance", ...}}
{"type": "function", "function": {"name": "block_card", ...}}
...
</tools>

If you choose to call a function ONLY reply in the following format...

You are a banking support agent named Oxen Banksy. Use tools when they are needed...
"""
```

The tool-calling format, the XML-style tags, the instructions -- all of this has to match the training data exactly. The model was fine-tuned to produce `<tool_call><function=...>` blocks, so the Modelfile needs to set up the same expectations.

---

## Step 6: Import into Ollama and Run

With the GGUF file and Modelfile ready, importing into Ollama is a one-liner:

```bash
cd models/ox-zesty-white-chipmunk-merged
ollama create ox-zesty-white-chipmunk -f Modelfile
```

That's it. Ollama reads the Modelfile, imports the quantized GGUF, and registers it under the name you provide.

Now you can run it:

```bash
ollama run ox-zesty-white-chipmunk
```

Or hit it via the API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "ox-zesty-white-chipmunk",
  "prompt": "What is my account balance for account ACC-12345?"
}'
```

---

## The One-Liner: Using the Conversion Script

If you don't want to run each step manually, the `convert_to_ollama.sh` script chains the entire pipeline together:

```bash
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk
```

That single command will:
1. Merge the LoRA adapter into the base model
2. Clone/update llama.cpp
3. Convert to BF16 GGUF
4. Quantize to Q4_K_M
5. Generate the Modelfile with system prompt and tools
6. Import into Ollama

You can customize the behavior with flags:

```bash
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk \
  --quant Q8_0 \
  --ollama-name my-banking-bot \
  --base-model Qwen/Qwen3.5-0.8B \
  --tools-path path/to/your/tools.json \
  --system-prompt path/to/your/system_prompt.txt
```

If you've already done some steps and want to skip ahead:

```bash
# Already merged? Skip straight to conversion
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk --skip-merge

# Already have the GGUF? Just generate the Modelfile and import
./scripts/convert_to_ollama.sh models/ox-zesty-white-chipmunk --skip-merge --skip-convert
```

---

## Troubleshooting

**"ollama is not installed"** -- Install with `brew install ollama` on macOS or grab it from [ollama.com](https://ollama.com).

**Model generates garbage or doesn't stop** -- The chat template probably doesn't match what the model was trained on. Double-check that the `TEMPLATE` in your Modelfile uses the right special tokens for your base model architecture.

**Out of memory during merge** -- The merge step loads the full base model into memory. For larger models (7B+), you may need 16-32GB of RAM. The script uses `device_map="cpu"` and `bfloat16` precision to keep memory usage reasonable.

**torch version conflicts** -- If you see errors about torch version mismatches after the llama.cpp setup, it's likely because llama.cpp's dependencies tried to pin a different torch version. The script avoids this by installing `gguf` with `--no-deps`, but if you ran `pip install -r requirements.txt` manually from the llama.cpp repo, your torch may have been downgraded. Reinstall with `pip install torch transformers peft accelerate`.

**Quantization quality** -- If tool calls are coming back malformed or the model seems confused, try a higher quantization level like Q6_K or Q8_0. Lower bit quantization can sometimes degrade structured output quality, especially for smaller models.

---

## Wrapping Up

The path from "I have LoRA weights on Oxen.ai" to "I'm running my model locally in Ollama" has a few steps, but each one is straightforward once you understand what's happening:

1. **Download** -- Pull your adapter weights from Oxen.ai
2. **Merge** -- Fold the LoRA deltas back into the base model
3. **Convert** -- Transform HuggingFace format into GGUF
4. **Quantize** -- Shrink the model for efficient local inference
5. **Configure** -- Generate a Modelfile that reproduces your training setup
6. **Import** -- Hand it to Ollama and start chatting

The most important thing to get right is the Modelfile -- especially the chat template and system prompt. The model learned to respond to very specific formatting during training, and if the inference-time setup doesn't match, you'll be wondering why your fine-tune seems broken when it's actually just getting the wrong prompts.

Happy inferencing.
