from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from train_industrial import WindowDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["data"]["manifest_root"]) / "fold_0"
    sources, origin_counts = set(), {}
    sample_count = 0
    for split in ("train", "val", "test"):
        dataset = WindowDataset(root / f"{split}.csv", cfg, train=False)
        for index in range(len(dataset)):
            video, _, meta = dataset[index]
            if tuple(video.shape) != (3, int(cfg["model"]["input_frames"]), 224, 224):
                raise RuntimeError(f"Unexpected tensor shape for {meta['sample_id']}: {tuple(video.shape)}")
            if not torch.isfinite(video).all():
                raise RuntimeError(f"Non-finite tensor for {meta['sample_id']}")
            sample_count += 1
            sources.add(meta["source_id"])
            origin = meta["sample_origin"]
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
    print({
        "input_mode": cfg["model"].get("input_mode", "full_frame"),
        "decoded_samples": sample_count,
        "source_count": len(sources),
        "sample_origins": origin_counts,
        "status": "ok",
    })


if __name__ == "__main__":
    main()

