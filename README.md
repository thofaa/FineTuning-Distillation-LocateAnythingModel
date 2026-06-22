# fine-tuning-code-for-low-luminance-detction

End-to-end training/deployment scaffolding for fine-tuning **...**
on low-light object detection data from ExDark using QLoRA, then merging/exporting to
GGUF for team deployment.

## Workflow

1. Parse ExDark JSON annotations into grounding prompts with normalized coordinates.
2. Fine-tune ... using QLoRA (4-bit), with validation each epoch.
3. Merge LoRA adapter into 16-bit base model and export to GGUF (`llama.cpp`).
4. Upload GGUF artifacts to Hugging Face Hub.
5. Run with Ollama or Dockerized runners.

## Repository Layout

coming soon ...

## Quick Start

coming soon ...
