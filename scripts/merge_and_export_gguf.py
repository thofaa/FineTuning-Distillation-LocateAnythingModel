#!/usr/bin/env python3
"""Merge QLoRA adapter into base model and export GGUF with llama.cpp."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_adapter(base_model: str, adapter_dir: Path, merged_out: Path) -> Path:
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    merged = peft_model.merge_and_unload()
    merged_out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out)
    return merged_out


def export_to_gguf(llama_cpp_dir: Path, merged_model_dir: Path, gguf_out: Path) -> None:
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    command = [
        "python",
        str(converter),
        str(merged_model_dir),
        "--outfile",
        str(gguf_out),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        stdout = exc.stdout or ""
        raise RuntimeError(
            f"GGUF export failed for command: {' '.join(command)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--merged-out", type=Path, required=True)
    parser.add_argument("--gguf-out", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    args = parser.parse_args()

    merged_dir = merge_adapter(args.base_model, args.adapter_dir, args.merged_out)
    export_to_gguf(args.llama_cpp_dir, merged_dir, args.gguf_out)
    print(f"Merged model exported to GGUF: {args.gguf_out}")


if __name__ == "__main__":
    main()
