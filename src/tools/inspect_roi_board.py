import argparse
from pathlib import Path
import csv
import sys

from PIL import Image, ImageDraw
import torch
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "industrial"
sys.path.insert(0, str(SCRIPT_ROOT))
from train_industrial import decode_window, read_normalized_roi, roi_transform  # noqa: E402

parser = argparse.ArgumentParser(description="Render ROI samples for board_drop inspection.")
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
manifest_root = Path(cfg["data"]["manifest_root"])
rows = []
for fold in range(4):
    with (manifest_root / f"fold_{fold}" / "test.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["class_name"] == "board_drop" and row["sample_origin"] == "anomaly_event":
                rows.append(row)

tiles = []
for row in sorted(rows, key=lambda item: item["source_id"]):
    decoded = decode_window(row["video_path"], float(row["window_start"]), float(row["window_end"]), 16)
    video = decoded.permute(3, 0, 1, 2).float().div_(255.0)
    cropped = roi_transform(video, read_normalized_roi(row["roi_path"]), False, cfg)
    cropped = (cropped * 0.225 + 0.45).clamp(0, 1)
    strip = Image.new("RGB", (224 * 4, 260), "white")
    for position, frame_index in enumerate((0, 5, 10, 15)):
        array = (cropped[:, frame_index].permute(1, 2, 0) * 255).byte().numpy()
        strip.paste(Image.fromarray(array), (position * 224, 0))
    ImageDraw.Draw(strip).text((8, 232), row["source_id"], fill="black")
    tiles.append(strip)

sheet = Image.new("RGB", (224 * 4, 260 * len(tiles)), "white")
for index, tile in enumerate(tiles):
    sheet.paste(tile, (0, index * 260))
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(output)
print(output)
