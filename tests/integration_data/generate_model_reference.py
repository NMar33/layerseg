"""Generate model reference data for weight/output tests.

Run from project root:
    python tests/integration_data/generate_model_reference.py
"""
import sys
import json
import hashlib
from pathlib import Path

import torch
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from binarizers.legacy import UNet, DoubleConv, Down, Up, OutConv
import __main__
for cls in [UNet, DoubleConv, Down, Up, OutConv]:
    setattr(__main__, cls.__name__, cls)

DATA_DIR = Path(__file__).parent


def main():
    model_path = PROJECT_ROOT / "pretrained_models" / "unet_220805.pth"
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()

    # 1. Weight info (checksums + shapes)
    state = model.state_dict()
    weight_info = {}
    for key, tensor in state.items():
        md5 = hashlib.md5(tensor.numpy().tobytes()).hexdigest()
        weight_info[key] = {
            "md5": md5,
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
        }

    info_path = DATA_DIR / "model_weight_info.json"
    with open(str(info_path), "w") as f:
        json.dump(weight_info, f, indent=2)
    print(f"Weight info: {info_path} ({len(weight_info)} keys)")

    # 2. Reference output on fixed input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y = model(x)

    out_path = DATA_DIR / "model_reference_output.pt"
    torch.save({"input": x, "output": y}, str(out_path))
    print(f"Reference output: {out_path}")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output mean:  {y.mean().item():.6f}")

    total_params = sum(v["numel"] for v in weight_info.values())
    print(f"\nTotal params: {total_params:,}")


if __name__ == "__main__":
    main()
