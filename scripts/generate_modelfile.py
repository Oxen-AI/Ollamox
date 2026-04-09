"""
Generate an Ollama Modelfile with the system prompt, tools, and chat template
baked in to match what the model was trained on.

Supports both Qwen (ChatML) and Gemma 4 template formats. The script auto-detects
the model family from --base-model and uses the correct template, stop tokens,
sampling parameters, and tool-definition format.

Usage:
    python scripts/generate_modelfile.py \
        --gguf-path ./model-Q4_K_M.gguf \
        --output Modelfile \
        --base-model Qwen/Qwen3.5-0.8B

    python scripts/generate_modelfile.py \
        --gguf-path ./model-Q4_K_M.gguf \
        --output Modelfile \
        --base-model google/gemma-4-E2B-it
"""

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _is_gemma4(base_model: str) -> bool:
    return "gemma-4" in base_model.lower()


# ---------------------------------------------------------------------------
# Qwen / ChatML format
# ---------------------------------------------------------------------------

def build_qwen_system_block(system_prompt: str, tools: list[dict]) -> str:
    """Build the system message content exactly as the Qwen3.5 chat template
    renders it when tools are provided."""
    parts = []
    parts.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
    for tool in tools:
        parts.append(json.dumps(tool))
    parts.append("</tools>")
    parts.append(
        "\nIf you choose to call a function ONLY reply in the following format "
        "with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner "
        "<function=...></function> block must be nested within "
        "<tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in "
        "natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like "
        "normal with your current knowledge and do not tell the user about "
        "function calls\n"
        "</IMPORTANT>"
    )
    parts.append("\n" + system_prompt.strip())
    return "\n".join(parts)


def build_qwen_modelfile(gguf_path: str, system_block: str | None = None) -> str:
    lines = []
    lines.append(f"FROM {gguf_path}")
    lines.append("")

    lines.append('TEMPLATE """')
    lines.append("{{- if .System }}<|im_start|>system")
    lines.append("{{ .System }}<|im_end|>")
    lines.append("{{ end }}")
    lines.append("{{- range .Messages }}<|im_start|>{{ .Role }}")
    lines.append("{{ .Content }}<|im_end|>")
    lines.append("{{ end }}<|im_start|>assistant")
    lines.append('"""')
    lines.append("")

    lines.append('PARAMETER stop "<|im_start|>"')
    lines.append('PARAMETER stop "<|im_end|>"')
    lines.append('PARAMETER stop "<|endoftext|>"')
    lines.append("")

    lines.append("PARAMETER temperature 0.7")
    lines.append("PARAMETER top_p 0.8")
    lines.append("PARAMETER repeat_penalty 1.05")
    lines.append("PARAMETER num_ctx 32768")

    if system_block:
        lines.append("")
        lines.append('SYSTEM """')
        lines.append(system_block)
        lines.append('"""')

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Gemma 4 format
# ---------------------------------------------------------------------------

def _tool_json_to_gemma4_declaration(tool: dict) -> str:
    """Convert an OpenAI-style tool JSON object to a Gemma 4 declaration.

    Input:  {"type": "function", "function": {"name": "get_balance",
             "parameters": {"properties": {"account_id": {"type": "string"}}}}}
    Output: <|tool>declaration:get_balance{account_id:string}<tool|>
    """
    func = tool.get("function", tool)
    name = func["name"]
    params = func.get("parameters", {}).get("properties", {})
    param_parts = []
    for pname, pspec in params.items():
        ptype = pspec.get("type", "string")
        param_parts.append(f"{pname}:{ptype}")
    param_str = ",".join(param_parts)
    return f"<|tool>declaration:{name}{{{param_str}}}<tool|>"


def build_gemma4_system_block(system_prompt: str, tools: list[dict]) -> str:
    """Build the system message content in Gemma 4's native format.

    Enables thinking mode and includes tool declarations using Gemma 4's
    <|tool>declaration:...<tool|> syntax.
    """
    parts = []
    parts.append("<|think|>")
    if system_prompt:
        parts.append(system_prompt.strip())
    for tool in tools:
        parts.append(_tool_json_to_gemma4_declaration(tool))
    return "\n".join(parts)


def build_gemma4_modelfile(gguf_path: str, system_block: str | None = None) -> str:
    lines = []
    lines.append(f"FROM {gguf_path}")
    lines.append("")

    # Gemma 4 uses <|turn>role ... <turn|> delimiters.
    # Ollama maps "assistant" -> "model" for Gemma.
    lines.append('TEMPLATE """')
    lines.append("{{- if .System }}<|turn>system")
    lines.append("{{ .System }}<turn|>")
    lines.append("{{ end }}")
    lines.append("{{- range .Messages }}")
    lines.append('{{- if eq .Role "user" }}<|turn>user')
    lines.append("{{ .Content }}<turn|>")
    lines.append('{{- else if eq .Role "assistant" }}<|turn>model')
    lines.append("{{ .Content }}<turn|>")
    lines.append("{{- end }}")
    lines.append("{{- end }}<|turn>model")
    lines.append('"""')
    lines.append("")

    # <turn|> ends the model's response; <tool_call|> stops for tool execution
    lines.append('PARAMETER stop "<turn|>"')
    lines.append('PARAMETER stop "<tool_call|>"')
    lines.append("")

    # Google's recommended sampling for Gemma 4
    lines.append("PARAMETER temperature 1.0")
    lines.append("PARAMETER top_p 0.95")
    lines.append("PARAMETER top_k 64")
    lines.append("PARAMETER num_ctx 32768")

    if system_block:
        lines.append("")
        lines.append('SYSTEM """')
        lines.append(system_block)
        lines.append('"""')

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def build_system_block(system_prompt: str, tools: list[dict], *, gemma4: bool = False) -> str:
    if gemma4:
        return build_gemma4_system_block(system_prompt, tools)
    return build_qwen_system_block(system_prompt, tools)


def build_modelfile(gguf_path: str, system_block: str | None = None, *, gemma4: bool = False) -> str:
    if gemma4:
        return build_gemma4_modelfile(gguf_path, system_block)
    return build_qwen_modelfile(gguf_path, system_block)


def main():
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile")
    parser.add_argument(
        "--gguf-path",
        type=str,
        required=True,
        help="Path to the GGUF file (used in the FROM directive)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the Modelfile",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="",
        help="HuggingFace base model ID (used to detect chat template format)",
    )
    parser.add_argument(
        "--tools-path",
        type=str,
        default=None,
        help="Path to tools.json (optional)",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default=None,
        help="Path to system_prompt.txt (optional)",
    )
    args = parser.parse_args()

    gemma4 = _is_gemma4(args.base_model)
    if gemma4:
        print(f"Detected Gemma 4 model — using Gemma 4 chat template and stop tokens")
    else:
        print(f"Using Qwen/ChatML chat template")

    tools = []
    if args.tools_path:
        with open(args.tools_path) as f:
            tools = json.load(f)

    system_prompt = ""
    if args.system_prompt_path:
        with open(args.system_prompt_path) as f:
            system_prompt = f.read()

    system_block = None
    if tools or system_prompt:
        system_block = build_system_block(system_prompt, tools, gemma4=gemma4)
    print(f"System block: {system_block}")

    modelfile = build_modelfile(args.gguf_path, system_block, gemma4=gemma4)
    print(f"Modelfile: {modelfile}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(modelfile)
    print(f"Modelfile written to: {output_path}")


if __name__ == "__main__":
    main()
