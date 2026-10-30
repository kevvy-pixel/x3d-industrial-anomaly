from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from train_x3d import build_model


def replace_swish(module: nn.Module) -> None:
    """Replace PyTorchVideo's custom autograd Swish with TorchScript-safe native SiLU."""
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "Swish":
            setattr(module, name, nn.SiLU(inplace=False))
        else:
            replace_swish(child)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(state["model"])
    model.eval()
    package = {
        "model": model.state_dict(), "classes": state["classes"],
        "metrics": state["metrics"], "config": state["config"],
        "input": {"layout": "NCTHW", "shape": [1, 3, 16, 224, 224],
                  "rgb_mean": [0.45, 0.45, 0.45], "rgb_std": [0.225, 0.225, 0.225]},
    }
    torch.save(package, args.output / "x3d_m_4class_inference.pt")
    replace_swish(model)
    example = torch.zeros(1, 3, 16, 224, 224)
    traced = torch.jit.trace(model, example, strict=False)
    with torch.inference_mode():
        max_error = float((model(example) - traced(example)).abs().max())
    if max_error > 1e-5:
        raise RuntimeError(f"TorchScript verification failed: max_error={max_error}")
    traced.save(str(args.output / "x3d_m_4class.ts"))
    (args.output / "inference_metadata.json").write_text(
        json.dumps({k: v for k, v in package.items() if k != "model"}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported inference checkpoint and TorchScript model; max_error={max_error:.3g}")


if __name__ == "__main__":
    main()
