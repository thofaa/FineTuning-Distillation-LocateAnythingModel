#!/usr/bin/env python3
"""Upload GGUF artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_artifact(repo_id: str, artifact: Path, token: str | None = None) -> None:
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact file not found: {artifact}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(artifact),
        path_in_repo=artifact.name,
        repo_id=repo_id,
        repo_type="model",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="Hugging Face model repo ID, e.g. org/model-name")
    parser.add_argument("--artifact", type=Path, required=True, help="Path to generated GGUF file")
    parser.add_argument("--token", help="Optional HF token. Otherwise use local cached authentication.")
    args = parser.parse_args()

    upload_artifact(repo_id=args.repo_id, artifact=args.artifact, token=args.token)
    print(f"Uploaded {args.artifact.name} to {args.repo_id}")


if __name__ == "__main__":
    main()
