import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "industrial"
sys.path.insert(0, str(SCRIPT_ROOT))
from train_industrial import WindowDataset

parser = argparse.ArgumentParser(description="Decode every sample in a generated fold.")
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--fold", type=int, default=0)
args = parser.parse_args()
cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
manifest = Path(cfg["data"]["manifest_root"]) / f"fold_{args.fold}"
for split in ("train", "val", "test"):
    dataset = WindowDataset(manifest / f"{split}.csv", cfg, train=False)
    for index, row in enumerate(dataset.rows):
        print(split, index, row["sample_id"], row["window_start"], row["window_end"], flush=True)
        video, _, _ = dataset[index]
        print("  OK", tuple(video.shape), flush=True)
print("ALL_OK", flush=True)
