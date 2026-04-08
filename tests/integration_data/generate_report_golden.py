"""Generate golden report data for report regression tests.

Run once from project root BEFORE library migration:
    python tests/integration_data/generate_report_golden.py

Produces golden_report/ with:
  - soft_mask.npy          raw float32 soft mask (before uint8 conversion)
  - soft_mask_uint8.png    mask as pipeline saves it
  - threshold_05.png       threshold mask t=0.5
  - threshold_07.png       threshold mask t=0.7
  - plot_preprocessed.png  matplotlib visualization (for manual review)
  - plot_segmentation.png  matplotlib visualization (for manual review)
  - report.pdf             PDF report (for manual review)
  - manifest.json          shapes, sizes, checksums, library versions
"""
import sys
import json
import hashlib
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from binarizers.legacy import UNet, DoubleConv, Down, Up, OutConv
import __main__
for cls in [UNet, DoubleConv, Down, Up, OutConv]:
    setattr(__main__, cls.__name__, cls)

from entities import BinarizerParams
from binarizer_pipeline import binarizer_pipeline
from utils import setup_logging

DATA_DIR = Path(__file__).parent
INPUTS_DIR = DATA_DIR / "inputs"
GOLDEN_DIR = DATA_DIR / "golden_report"

# Use real_crop.png as golden input (128x128, already exists)
GOLDEN_INPUT = "real_crop.png"


def generate():
    print("Generating golden report data...")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Verify input exists
    input_path = INPUTS_DIR / GOLDEN_INPUT
    assert input_path.exists(), "Input %s not found. Run generate_reference.py first." % input_path

    setup_logging("default")

    with tempfile.TemporaryDirectory() as tmp_out:
        # Create a temp input dir with just the one image
        tmp_input = Path(tmp_out) / "input"
        tmp_input.mkdir()
        shutil.copy2(str(input_path), str(tmp_input / GOLDEN_INPUT))

        tmp_report = Path(tmp_out) / "reports"
        tmp_report.mkdir()

        params = BinarizerParams(
            path_imgs_dir=str(tmp_input),
            path_report_dir=str(tmp_report),
            path_logging_config="default",
            path_models_dir=str(PROJECT_ROOT / "pretrained_models"),
            model_name="unet_220805.pth",
            cache=True,
            cache_dir=str(Path(tmp_out) / "cache"),
            report_name="golden",
            scale_factors=[1],
            gaussian_blur=False,
            gaussian_blur_kernel_size=5,
            binarizer_thresholds=[0.5, 0.7],
            original_img_color_map="gray",
            imgs_in_row=3,
            color_interest="black",
            report_dpi=150,
            report_fig_sz=4,
            report_short=False,
            short_report_dir=str(Path(tmp_out) / "short"),
            device="cpu",
        )

        print("  Running pipeline...")
        binarizer_pipeline(params)

        # Find generated report directory
        report_dirs = [d for d in tmp_report.iterdir() if d.is_dir()]
        assert len(report_dirs) == 1, "Expected 1 report dir, got %d" % len(report_dirs)
        report_dir = report_dirs[0]

        masks_dir = report_dir / "masks"
        assert masks_dir.exists(), "masks/ not found in %s" % report_dir

        # Extract masks
        mask_files = sorted(masks_dir.glob("*.png"))
        print("  Found %d mask files" % len(mask_files))

        for mf in mask_files:
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            name = mf.name
            if "soft_bin" in name:
                cv2.imwrite(str(GOLDEN_DIR / "soft_mask_uint8.png"), mask)
                # Also save as float for higher-precision comparison
                np.save(str(GOLDEN_DIR / "soft_mask_float.npy"), mask.astype(np.float32) / 255.0)
                print("  soft_mask_uint8.png: %s, range [%d, %d]" % (
                    str(mask.shape), mask.min(), mask.max()))
            elif "threshold_0.5" in name:
                cv2.imwrite(str(GOLDEN_DIR / "threshold_05.png"), mask)
                print("  threshold_05.png: %s, unique=%s" % (
                    str(mask.shape), str(np.unique(mask))))
            elif "threshold_0.7" in name:
                cv2.imwrite(str(GOLDEN_DIR / "threshold_07.png"), mask)
                print("  threshold_07.png: %s, unique=%s" % (
                    str(mask.shape), str(np.unique(mask))))

        # Copy visualization PNGs (for manual review, not pixel-tested)
        viz_files = sorted(report_dir.glob("*.png"))
        for i, vf in enumerate(viz_files):
            dst_name = "viz_%02d_%s" % (i, vf.name)
            shutil.copy2(str(vf), str(GOLDEN_DIR / dst_name))
            print("  %s" % dst_name)

        # Copy PDF
        pdf_files = list(tmp_report.glob("*.pdf"))
        for pf in pdf_files:
            shutil.copy2(str(pf), str(GOLDEN_DIR / "report.pdf"))
            print("  report.pdf: %d bytes" % pf.stat().st_size)

        # Save directory structure info
        structure = {
            "report_dir_name": report_dir.name,
            "files_in_report_dir": sorted([f.name for f in report_dir.iterdir()]),
            "files_in_masks_dir": sorted([f.name for f in masks_dir.iterdir()]),
            "pdf_files": sorted([f.name for f in tmp_report.glob("*.pdf")]),
        }

        # Manifest
        manifest = {
            "input_image": GOLDEN_INPUT,
            "params": {
                "scale_factors": [1],
                "gaussian_blur": False,
                "thresholds": [0.5, 0.7],
                "color_interest": "black",
                "report_dpi": 150,
                "report_fig_sz": 4,
            },
            "library_versions": {
                "cv2": cv2.__version__,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version,
            },
            "structure": structure,
            "golden_files": {},
        }

        for f in sorted(GOLDEN_DIR.iterdir()):
            if f.name == "manifest.json":
                continue
            data = f.read_bytes()
            manifest["golden_files"][f.name] = {
                "size": len(data),
                "md5": hashlib.md5(data).hexdigest(),
            }

        with open(str(GOLDEN_DIR / "manifest.json"), "w") as fp:
            json.dump(manifest, fp, indent=2)

        print("\nDone! Golden report data in %s" % GOLDEN_DIR)
        print("Files: %d" % len(manifest["golden_files"]))


if __name__ == "__main__":
    generate()
