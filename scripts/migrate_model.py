"""Migrate model from joblib format to PyTorch state_dict.

Run from project root:
    python scripts/migrate_model.py
"""
import sys
import hashlib
from pathlib import Path

import torch
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from binarizers.legacy import UNet, DoubleConv, Down, Up, OutConv
import __main__
for cls in [UNet, DoubleConv, Down, Up, OutConv]:
    setattr(__main__, cls.__name__, cls)


def main():
    src = PROJECT_ROOT / "pretrained_models" / "unet_220805.model"
    dst = PROJECT_ROOT / "pretrained_models" / "unet_220805.pth"

    print(f"Loading joblib model: {src}")
    model_old = joblib.load(src.as_posix())
    model_old.eval()
    state = model_old.state_dict()
    print(f"  Keys: {len(state)}, Params: {sum(p.numel() for p in state.values()):,}")

    # Save as state_dict
    torch.save(state, str(dst))
    print(f"Saved state_dict: {dst} ({dst.stat().st_size / 1024 / 1024:.1f} MB)")

    # Verify: load into fresh UNet and compare
    model_new = UNet(n_channels=3, n_classes=2, bilinear=False)
    model_new.load_state_dict(torch.load(str(dst)))
    model_new.eval()

    # Compare outputs
    torch.manual_seed(42)
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y_old = model_old(x)
        y_new = model_new(x)

    if torch.equal(y_old, y_new):
        print("Verification: outputs are IDENTICAL")
    else:
        max_diff = (y_old - y_new).abs().max().item()
        print(f"WARNING: outputs differ. Max diff: {max_diff}")

    # Compare checksums
    state_new = model_new.state_dict()
    mismatches = 0
    for key in state:
        old_md5 = hashlib.md5(state[key].numpy().tobytes()).hexdigest()
        new_md5 = hashlib.md5(state_new[key].numpy().tobytes()).hexdigest()
        if old_md5 != new_md5:
            print(f"  MISMATCH: {key}")
            mismatches += 1

    if mismatches == 0:
        print("Verification: all weight checksums MATCH")
    else:
        print(f"WARNING: {mismatches} checksum mismatches")


if __name__ == "__main__":
    main()
