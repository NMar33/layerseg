"""Generate reference data for library compatibility tests.

Run once from project root BEFORE library migration:
    python tests/integration_data/generate_library_references.py

All input values are deterministic (no random), so reference is reproducible
on the same library versions.
"""
import sys
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from binarizers.legacy import UNet, DoubleConv, Down, Up, OutConv
import __main__
for cls in [UNet, DoubleConv, Down, Up, OutConv]:
    setattr(__main__, cls.__name__, cls)

import torchvision.transforms as T
from binarizers.seg_model import seg_post_m220805

DATA_DIR = Path(__file__).parent

# --- Deterministic input arrays ---

def _make_uint8_4x4():
    """Deterministic 4x4 uint8 array with known values."""
    return np.array([
        [10, 50, 100, 200],
        [20, 60, 110, 210],
        [30, 70, 120, 220],
        [40, 80, 130, 230],
    ], dtype=np.uint8)


def _make_float32_4x4():
    """Deterministic 4x4 float32 array."""
    return np.array([
        [0.1, 0.25, 0.5, 0.75],
        [0.15, 0.3, 0.55, 0.8],
        [0.2, 0.35, 0.6, 0.85],
        [0.0, 0.4, 0.65, 1.0],
    ], dtype=np.float32)


def _make_uint8_8x8():
    """Deterministic 8x8 uint8 array for GaussianBlur."""
    row = np.arange(0, 256, 32, dtype=np.uint8)  # [0, 32, 64, ..., 224]
    return np.tile(row, (8, 1))


def _make_sigmoid_logits():
    """Deterministic 2-channel logits for seg_post_m220805."""
    ch0 = np.array([[0.5, 1.0, -0.5], [2.0, 0.0, -1.0]], dtype=np.float32)
    ch1 = np.array([[1.0, 0.5, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
    return torch.tensor(np.stack([ch0, ch1]))


def _make_float32_mask_16x16():
    """Deterministic 16x16 float32 mask for resize test."""
    row = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    return np.outer(row, row)


# --- Generation ---

def generate():
    print("Generating library reference data...")

    # 1. cv2.resize uint8: 4x4 -> 8x8
    img_u8 = _make_uint8_4x4()
    resized_u8 = cv2.resize(img_u8, (8, 8))
    np.save(str(DATA_DIR / "cv2_resize_uint8_ref.npy"), resized_u8)
    print("  cv2_resize_uint8_ref.npy: %s" % str(resized_u8.shape))

    # 2. cv2.resize float32: 4x4 -> 8x8
    img_f32 = _make_float32_4x4()
    resized_f32 = cv2.resize(img_f32, (8, 8))
    np.save(str(DATA_DIR / "cv2_resize_float32_ref.npy"), resized_f32)
    print("  cv2_resize_float32_ref.npy: %s" % str(resized_f32.shape))

    # 3. cv2.GaussianBlur: 8x8 -> blur(5,5)
    img_blur = _make_uint8_8x8()
    blurred = cv2.GaussianBlur(img_blur, (5, 5), 0)
    np.save(str(DATA_DIR / "cv2_gaussian_blur_ref.npy"), blurred)
    print("  cv2_gaussian_blur_ref.npy: %s" % str(blurred.shape))

    # 4. torchvision ToTensor: uint8 -> float32
    img_tt = _make_uint8_4x4()
    tensor_tt = T.ToTensor()(img_tt)
    torch.save(tensor_tt, str(DATA_DIR / "to_tensor_ref.pt"))
    print("  to_tensor_ref.npy: %s" % str(tensor_tt.shape))

    # 5. F.pad reference
    inp_pad = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    padded = F.pad(inp_pad.unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0).squeeze(0)
    torch.save(padded, str(DATA_DIR / "fpad_ref.pt"))
    print("  fpad_ref.pt: %s" % str(padded.shape))

    # 6. sigmoid post-processing reference
    logits = _make_sigmoid_logits()
    sigmoid_out = seg_post_m220805(logits)
    np.save(str(DATA_DIR / "sigmoid_post_ref.npy"), sigmoid_out)
    print("  sigmoid_post_ref.npy: %s" % str(sigmoid_out.shape))

    # 7. cv2.resize float32 mask: 16x16 -> 8x8
    mask_f32 = _make_float32_mask_16x16()
    resized_mask = cv2.resize(mask_f32, (8, 8))
    np.save(str(DATA_DIR / "seg_resize_ref.npy"), resized_mask)
    print("  seg_resize_ref.npy: %s" % str(resized_mask.shape))

    # 8. scale_factor_prep reference: 10x10 uint8, sf=2.0
    img_sf = np.arange(100, dtype=np.uint8).reshape(10, 10)
    resized_sf = cv2.resize(img_sf, (20, 20))
    np.save(str(DATA_DIR / "scale_factor_prep_ref.npy"), resized_sf)
    print("  scale_factor_prep_ref.npy: %s" % str(resized_sf.shape))

    # 9. Manifest with checksums and library versions
    manifest = {
        "cv2_version": cv2.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "files": {}
    }
    for f in DATA_DIR.glob("*_ref.*"):
        data = f.read_bytes()
        manifest["files"][f.name] = {
            "size": len(data),
            "md5": hashlib.md5(data).hexdigest()
        }
    with open(str(DATA_DIR / "library_ref_manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)
    print("  library_ref_manifest.json written")

    print("\nDone! %d reference files generated." % len(manifest["files"]))


if __name__ == "__main__":
    generate()
