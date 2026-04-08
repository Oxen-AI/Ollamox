"""
Generate an Ollama Modelfile with the system prompt, tools, and chat template
baked in to match what the model was trained on.

Usage:
    python scripts/generate_modelfile.py \
        --gguf-path ./model-Q4_K_M.gguf \
        --output Modelfile

The tool definitions and system prompt are read from data/tools.json and
data/system_prompt.txt respectively, then formatted in the same way the
Qwen3.5 chat template renders them during training.
"""

import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


def build_system_block(system_prompt: str, tools: list[dict]) -> str:
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


def build_modelfile(gguf_path: str, system_block: str | None = None) -> str:
    """Build the full Ollama Modelfile content."""
    lines = []
    lines.append(f"FROM {gguf_path}")
    lines.append("")

    # Chat template: simple ChatML that matches Qwen3.5 training format.
    # The tool instructions and definitions are already in the SYSTEM block,
    # so the template only needs to handle message roles.
    lines.append('TEMPLATE """')
    lines.append("{{- if .System }}<|im_start|>system")
    lines.append("{{ .System }}<|im_end|>")
    lines.append("{{ end }}")
    lines.append("{{- range .Messages }}<|im_start|>{{ .Role }}")
    lines.append("{{ .Content }}<|im_end|>")
    lines.append("{{ end }}<|im_start|>assistant")
    lines.append('"""')
    lines.append("")

    # Stop tokens
    lines.append('PARAMETER stop "<|im_start|>"')
    lines.append('PARAMETER stop "<|im_end|>"')
    lines.append('PARAMETER stop "<|endoftext|>"')
    lines.append("")

    # Generation parameters
    lines.append("PARAMETER temperature 0.7")
    lines.append("PARAMETER top_p 0.8")
    lines.append("PARAMETER repeat_penalty 1.05")
    lines.append("PARAMETER num_ctx 32768")

    # System prompt with tools baked in (only if provided)
    if system_block:
        lines.append("")
        lines.append('SYSTEM """')
        lines.append(system_block)
        lines.append('"""')

    return "\n".join(lines) + "\n"


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
        system_block = build_system_block(system_prompt, tools)

    modelfile = build_modelfile(args.gguf_path, system_block)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(modelfile)
    print(f"Modelfile written to: {output_path}")


if __name__ == "__main__":
    main()
