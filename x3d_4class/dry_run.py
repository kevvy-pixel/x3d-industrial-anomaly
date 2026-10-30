from pathlib import Path
import csv
import sys

import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from train_x3d import ClipDataset, build_model

with (PROJECT / "experiment" / "samples.csv").open("r", encoding="utf-8-sig", newline="") as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
clips = []
targets = []
for row in rows[:2]:
    clip, target, _ = ClipDataset([row], train=True)[0]
    clips.append(clip)
    targets.append(target)
x = torch.stack(clips).cuda()
y = torch.tensor(targets, device="cuda")
model = build_model(pretrained=True).cuda().train()
with torch.amp.autocast("cuda"):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
print({"input": tuple(x.shape), "output": tuple(logits.shape), "loss": float(loss),
       "allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1)})
