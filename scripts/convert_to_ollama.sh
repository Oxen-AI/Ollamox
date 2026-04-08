#!/usr/bin/env bash
#
# Convert a merged Qwen3.5 model to GGUF and import into Ollama.
#
# Prerequisites:
#   pip install torch transformers peft accelerate
#   brew install ollama   (or install from https://ollama.com)
#
# Usage:
#   ./scripts/convert_to_ollama.sh <model-path> [OPTIONS]
#
# Arguments:
#   model-path              Path to the LoRA adapter directory (required)
#
# Options:
#   --merged-path PATH      Path for merged model output (default: <model-path>-merged)
#   --base-model MODEL      HuggingFace base model ID (default: auto-detect from adapter config)
#   --quant TYPE            Quantization type (default: Q4_K_M)
#   --ollama-name NAME      Name for the Ollama model (default: basename of model-path)
#   --tools-path PATH       Path to tools.json (default: data/tools.json)
#   --system-prompt PATH    Path to system_prompt.txt (default: data/system_prompt.txt)
#   --skip-merge            Skip LoRA merge step (use if already merged)
#   --skip-convert          Skip GGUF conversion (use if already converted)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
ADAPTER_PATH=""
MERGED_PATH=""
BASE_MODEL=""
QUANT_TYPE="Q4_K_M"
OLLAMA_NAME=""
TOOLS_PATH=""
SYSTEM_PROMPT_PATH=""
SKIP_MERGE=false
SKIP_CONVERT=false
LLAMA_CPP_DIR="${PROJECT_DIR}/tools/llama.cpp"

usage() {
    echo "Usage: $0 <model-path> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  model-path              Path to the LoRA adapter directory (required)"
    echo ""
    echo "Options:"
    echo "  --merged-path PATH      Path for merged model output (default: <model-path>-merged)"
    echo "  --base-model MODEL      HuggingFace base model ID (default: auto-detect)"
    echo "  --quant TYPE            Quantization type (default: Q4_K_M)"
    echo "  --ollama-name NAME      Name for the Ollama model (default: basename of model-path)"
    echo "  --tools-path PATH       Path to tools.json (default: data/tools.json)"
    echo "  --system-prompt PATH    Path to system_prompt.txt (default: data/system_prompt.txt)"
    echo "  --skip-merge            Skip LoRA merge step"
    echo "  --skip-convert          Skip GGUF conversion step"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --merged-path) MERGED_PATH="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --quant) QUANT_TYPE="$2"; shift 2 ;;
        --ollama-name) OLLAMA_NAME="$2"; shift 2 ;;
        --tools-path) TOOLS_PATH="$2"; shift 2 ;;
        --system-prompt) SYSTEM_PROMPT_PATH="$2"; shift 2 ;;
        --skip-merge) SKIP_MERGE=true; shift ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        --help|-h) usage ;;
        -*)  echo "Unknown option: $1"; usage ;;
        *)
            if [ -z "$ADAPTER_PATH" ]; then
                ADAPTER_PATH="$1"; shift
            else
                echo "Unexpected argument: $1"; usage
            fi
            ;;
    esac
done

# Require model path
if [ -z "$ADAPTER_PATH" ]; then
    echo "Error: model path is required."
    echo ""
    usage
fi

# Resolve to absolute path
ADAPTER_PATH="$(cd "$ADAPTER_PATH" && pwd)"

# Derive defaults from the adapter path
if [ -z "$MERGED_PATH" ]; then
    MERGED_PATH="${ADAPTER_PATH}-merged"
fi
if [ -z "$OLLAMA_NAME" ]; then
    OLLAMA_NAME="$(basename "$ADAPTER_PATH")"
fi

GGUF_BF16="${MERGED_PATH}/model-bf16.gguf"
GGUF_QUANTIZED="${MERGED_PATH}/model-${QUANT_TYPE}.gguf"
MODELFILE="${MERGED_PATH}/Modelfile"

echo "============================================"
echo "  LoRA -> GGUF -> Ollama Conversion Pipeline"
echo "============================================"
echo ""
echo "  Adapter path:   ${ADAPTER_PATH}"
echo "  Merged path:    ${MERGED_PATH}"
echo "  Quantization:   ${QUANT_TYPE}"
echo "  Ollama name:    ${OLLAMA_NAME}"
echo "  Tools:          ${TOOLS_PATH:-none}"
echo "  System prompt:  ${SYSTEM_PROMPT_PATH:-none}"
echo ""

# -----------------------------------------------
# Step 1: Merge LoRA into base model
# -----------------------------------------------
if [ "$SKIP_MERGE" = false ]; then
    echo ">>> Step 1: Merging LoRA adapter into base model..."
    MERGE_ARGS=(
        --adapter-path "$ADAPTER_PATH"
        --output-path "$MERGED_PATH"
    )
    if [ -n "$BASE_MODEL" ]; then
        MERGE_ARGS+=(--base-model "$BASE_MODEL")
    fi
    python "${SCRIPT_DIR}/merge_lora.py" "${MERGE_ARGS[@]}"
    echo ""
else
    echo ">>> Step 1: Skipping merge (--skip-merge)"
    echo ""
fi

# -----------------------------------------------
# Step 2: Clone/update llama.cpp and convert to GGUF
# -----------------------------------------------
if [ "$SKIP_CONVERT" = false ]; then
    echo ">>> Step 2a: Setting up llama.cpp..."

    if [ ! -d "$LLAMA_CPP_DIR" ]; then
        echo "Cloning llama.cpp..."
        mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    else
        echo "llama.cpp already exists at ${LLAMA_CPP_DIR}, pulling latest..."
        git -C "$LLAMA_CPP_DIR" pull --ff-only || true
    fi

    # Install only the Python packages needed by convert_hf_to_gguf.py.
    # Do NOT use llama.cpp's requirements.txt — it pins torch~=2.6.0 which
    # will downgrade torch and break peft/transformers/torchvision compatibility.
    # Use --no-deps to prevent any transitive dependency from touching torch.
    echo "Installing conversion script dependencies..."
    pip install --no-deps gguf
    pip install numpy sentencepiece protobuf pyyaml tqdm

    # Build quantize tool
    echo ""
    echo ">>> Step 2b: Building llama.cpp quantize tool..."
    cmake -S "$LLAMA_CPP_DIR" -B "${LLAMA_CPP_DIR}/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${LLAMA_CPP_DIR}/build" --target llama-quantize -j "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

    # Convert to BF16 GGUF
    echo ""
    echo ">>> Step 2c: Converting merged model to BF16 GGUF..."
    python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "$MERGED_PATH" \
        --outtype bf16 \
        --outfile "$GGUF_BF16"

    echo "BF16 GGUF saved to: ${GGUF_BF16}"

    # Quantize
    echo ""
    echo ">>> Step 2d: Quantizing to ${QUANT_TYPE}..."
    "${LLAMA_CPP_DIR}/build/bin/llama-quantize" \
        "$GGUF_BF16" \
        "$GGUF_QUANTIZED" \
        "$QUANT_TYPE"

    echo "Quantized GGUF saved to: ${GGUF_QUANTIZED}"
    echo ""
else
    echo ">>> Step 2: Skipping GGUF conversion (--skip-convert)"
    echo ""
fi

# -----------------------------------------------
# Step 3: Create Ollama Modelfile and import
# -----------------------------------------------
echo ">>> Step 3: Creating Ollama Modelfile..."

MODELFILE_ARGS=(
    --gguf-path "./model-${QUANT_TYPE}.gguf"
    --output "$MODELFILE"
)
if [ -n "$TOOLS_PATH" ]; then
    MODELFILE_ARGS+=(--tools-path "$TOOLS_PATH")
fi
if [ -n "$SYSTEM_PROMPT_PATH" ]; then
    MODELFILE_ARGS+=(--system-prompt-path "$SYSTEM_PROMPT_PATH")
fi
python "${SCRIPT_DIR}/generate_modelfile.py" "${MODELFILE_ARGS[@]}"

echo ""

echo ">>> Step 4: Importing into Ollama..."

# Check that Ollama is available
if ! command -v ollama &> /dev/null; then
    echo "ERROR: ollama is not installed."
    echo "Install it with: brew install ollama"
    echo ""
    echo "After installing, run:"
    echo "  cd ${MERGED_PATH} && ollama create ${OLLAMA_NAME} -f Modelfile"
    exit 1
fi

# Create the model in Ollama (run from the merged dir so relative path in Modelfile works)
cd "$MERGED_PATH"
ollama create "$OLLAMA_NAME" -f Modelfile

echo ""
echo "============================================"
echo "  Done!"
echo "============================================"
echo ""
echo "  Run your model with:"
echo "    ollama run ${OLLAMA_NAME}"
echo ""
echo "  Or use the API:"
echo "    curl http://localhost:11434/api/generate -d '{"
echo "      \"model\": \"${OLLAMA_NAME}\","
echo "      \"prompt\": \"Hello!\""
echo "    }'"
echo ""
