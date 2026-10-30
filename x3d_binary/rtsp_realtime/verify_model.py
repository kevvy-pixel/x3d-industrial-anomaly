from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the packaged checkpoint and run one dummy forward pass.")
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    model_path = root / metadata["checkpoint_file"]
    actual_hash = sha256(model_path)
    if actual_hash != metadata["checkpoint_sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {actual_hash}")

    print(
        f"model hash OK: epoch={metadata['selected_epoch']} "
        f"classes={metadata['classes']} sha256={actual_hash}"
    )

    if args.skip_forward:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device).eval()
    dummy = torch.zeros(1, 3, metadata["input_frames"], 224, 224, device=device)
    with torch.inference_mode():
        output = model(dummy)
    if tuple(output.shape) != (1, len(metadata["classes"])):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    print(f"forward OK: device={device} output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
