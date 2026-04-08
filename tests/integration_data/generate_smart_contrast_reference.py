"""Generate reference data for smart_contrast optimization tests.

Run from project root:
    python tests/integration_data/generate_smart_contrast_reference.py

Saves: tests/integration_data/smart_contrast_reference.pt
Contains: input tensor + output of make_img_with_smart_layers +
          intermediate contrast_3x3 and contrast_7x7.

Re-run only if intentionally changing smart_contrast behavior.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from binarizers.legacy import make_smart_contrast, make_img_with_smart_layers

DATA_DIR = Path(__file__).parent


def main():
    torch.manual_seed(42)
    img = torch.rand(1, 16, 16)

    print("Input shape:", img.shape)
    print("Input mean:", img.mean().item())

    contrast_3x3 = make_smart_contrast(img, (3, 3))
    contrast_7x7 = make_smart_contrast(1.0 - img, (7, 7))
    full_output = make_img_with_smart_layers(img)

    out_path = DATA_DIR / "smart_contrast_reference.pt"
    torch.save({
        "input": img,
        "contrast_3x3": contrast_3x3,
        "contrast_7x7": contrast_7x7,
        "output": full_output,
    }, str(out_path))

    print(f"Saved: {out_path}")
    print(f"  contrast_3x3 NaN count: {contrast_3x3.isnan().sum().item()}")
    print(f"  contrast_7x7 NaN count: {contrast_7x7.isnan().sum().item()}")
    print(f"  output shape: {full_output.shape}")
    print(f"  output NaN count: {full_output.isnan().sum().item()}")


if __name__ == "__main__":
    main()
