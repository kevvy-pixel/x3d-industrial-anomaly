from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_touchface_binary import build_binary_model


def replace_swish(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Swish":
            setattr(module, name, nn.SiLU(inplace=False))
        else:
            replace_swish(child)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    model = build_binary_model(args.backbone)
    model.load_state_dict(state["model"], strict=True)
    model.eval(); replace_swish(model)
    example = torch.zeros(1, 3, 16, 224, 224)
    traced = torch.jit.trace(model, example, strict=False)
    with torch.inference_mode():
        error = float((model(example) - traced(example)).abs().max())
    if error > 1e-5:
        raise RuntimeError(f"TorchScript mismatch {error}")
    output_path = args.output / "x3d_m_normal_touch_face.ts"
    traced.save(str(output_path))
    metadata = {"classes": list(state["classes"]), "metrics": state["metrics"], "config": state["config"], "input": {"layout": "NCTHW", "shape": [1, 3, 16, 224, 224], "mean": [0.45] * 3, "std": [0.225] * 3}, "torchscript_max_error": error}
    (args.output / "inference_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "max_error": error}, ensure_ascii=False))


if __name__ == "__main__":
    main()
