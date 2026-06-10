# low-luminance-detection-training---quantized-qwen3-vl

End-to-end training/deployment scaffolding for fine-tuning **Qwen3-VL-8B-Instruct**
on low-light object detection data from ExDark using QLoRA, then merging/exporting to
GGUF for team deployment.

## Workflow

1. Parse ExDark JSON annotations into grounding prompts with normalized coordinates.
2. Fine-tune Qwen3-VL-8B-Instruct using QLoRA (4-bit), with validation each epoch.
3. Merge LoRA adapter into 16-bit base model and export to GGUF (`llama.cpp`).
4. Upload GGUF artifacts to Hugging Face Hub.
5. Run with Ollama or Dockerized runners.

## Repository Layout

- `scripts/preprocess_exdark.py` – ExDark JSON → normalized grounding prompts.
- `scripts/train_qlora_colab.py` – Colab-friendly QLoRA training entrypoint.
- `scripts/merge_and_export_gguf.py` – merge adapters + GGUF conversion wrapper.
- `scripts/upload_gguf_to_hf.py` – upload GGUF outputs to Hugging Face Hub.
- `deployment/Modelfile` – Ollama model config template.
- `deployment/docker-compose.yml` – Docker runtime example.
- `tests/test_preprocess_exdark.py` – focused parser normalization tests.

## Quick Start

```bash
# 1) Parse annotations
python scripts/preprocess_exdark.py \
  --input /path/to/annotations.json \
  --output /path/to/train_prompts.jsonl \
  --image-width 1920 \
  --image-height 1080

# 2) Train (Google Colab recommended)
python scripts/train_qlora_colab.py \
  --train-jsonl /path/to/train_prompts.jsonl \
  --val-jsonl /path/to/val_prompts.jsonl \
  --output-dir /path/to/lora-output

# 3) Merge + export GGUF
python scripts/merge_and_export_gguf.py \
  --base-model Qwen/Qwen3-VL-8B-Instruct \
  --adapter-dir /path/to/lora-output \
  --merged-out /path/to/merged-model \
  --gguf-out /path/to/model.gguf \
  --llama-cpp-dir /path/to/llama.cpp

# 4) Upload GGUF
python scripts/upload_gguf_to_hf.py \
  --repo-id your-org/your-model-repo \
  --artifact /path/to/model.gguf
```