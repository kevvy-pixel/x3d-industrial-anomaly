from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

if os.name == "nt":
    cuda_bin = str(Path(sys.executable).resolve().parent / "bin")
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    os.add_dll_directory(cuda_bin)
import torch

from train_x3d import ClipDataset


ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiment"
with (EXP / "samples.csv").open("r", encoding="utf-8-sig", newline="") as f:
    row = next(r for r in csv.DictReader(f) if r["split"] == "val" and r["label"] != "normal")
clip, target, sample_id = ClipDataset([row], train=False)[0]
device = torch.device("cuda")
model = torch.jit.load(str(EXP / "x3d_m_4class.ts"), map_location=device).eval()
x = clip.unsqueeze(0).to(device)
for _ in range(3):
    with torch.inference_mode():
        model(x)
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    logits = model(x)
torch.cuda.synchronize()
latency_ms = (time.perf_counter() - started) * 1000
probs = torch.softmax(logits.float(), 1)[0].cpu()
result = {"sample_id": sample_id, "truth": row["label"], "prediction": ("normal", "single_hand", "board_drop", "touch_face")[int(probs.argmax())],
          "confidence": float(probs.max()), "latency_ms": latency_ms, "probabilities": probs.tolist()}
(EXP / "deployment_smoke_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
