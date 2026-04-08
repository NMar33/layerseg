"""Generate reference data for integration tests.

Run once from project root:
    python tests/integration_data/generate_reference.py

This script:
1. Prepares 5 input images (128x128) from test_dataset + assessment synthetic
2. Runs the binarizer pipeline on each (scale=[1], no_blur, thresholds=[0.5, 0.7])
3. Extracts soft mask + threshold masks → saves to expected/
4. Copies GT masks → saves to ground_truth/

Commit the results to git. Re-run only when intentionally changing model behavior.
"""
import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Register UNet classes in __main__ for joblib
from binarizers.legacy import UNet, DoubleConv, Down, Up, OutConv
import __main__
for cls in [UNet, DoubleConv, Down, Up, OutConv]:
    setattr(__main__, cls.__name__, cls)

from entities import BinarizerParams
from binarizer_pipeline import binarizer_pipeline
from utils import setup_logging

DATA_DIR = Path(__file__).parent
INPUTS_DIR = DATA_DIR / "inputs"
EXPECTED_DIR = DATA_DIR / "expected"
GT_DIR = DATA_DIR / "ground_truth"

# Source images to collect
SOURCES = [
    {
        "name": "real_crop",
        "img_src": PROJECT_ROOT / "assessment" / "test_dataset" / "real" / "imgs" / "16_1.tif",
        "gt_src": PROJECT_ROOT / "assessment" / "output" / "test_dataset_gt" / "real" / "16_1.png",
        "crop": (0, 0, 128, 128),  # y, x, h, w
    },
    {
        "name": "synth_type02_crop",
        "img_src": PROJECT_ROOT / "assessment" / "output" / "test_dataset_imgs" / "synthetic" / "type_02_500_001.png",
        "gt_src": PROJECT_ROOT / "assessment" / "output" / "test_dataset_gt" / "synthetic" / "type_02_500_001.png",
        "crop": (100, 100, 128, 128),
    },
    {
        "name": "wavy_thin_gray_bg_128x128",
        "img_src": PROJECT_ROOT / "assessment" / "output" / "synthetic_imgs" / "wavy_lines" / "wavy_thin_gray_bg_128x128.png",
        "gt_src": PROJECT_ROOT / "assessment" / "output" / "ground_truth" / "wavy_lines" / "wavy_thin_gray_bg_128x128.png",
    },
    {
        "name": "horiz_uniform_128x128",
        "img_src": PROJECT_ROOT / "assessment" / "output" / "synthetic_imgs" / "straight_lines" / "horiz_uniform_128x128.png",
        "gt_src": PROJECT_ROOT / "assessment" / "output" / "ground_truth" / "straight_lines" / "horiz_uniform_128x128.png",
    },
    {
        "name": "checker_uniform_128x128",
        "img_src": PROJECT_ROOT / "assessment" / "output" / "synthetic_imgs" / "checkerboard" / "checker_uniform_128x128.png",
        "gt_src": PROJECT_ROOT / "assessment" / "output" / "ground_truth" / "checkerboard" / "checker_uniform_128x128.png",
    },
]


def prepare_inputs():
    """Copy/crop source images into inputs/ and ground_truth/."""
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    for src in SOURCES:
        name = src["name"]
        img = cv2.imread(str(src["img_src"]), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Cannot read {src['img_src']}"

        if "crop" in src:
            y, x, h, w = src["crop"]
            img = img[y:y+h, x:x+w]

        cv2.imwrite(str(INPUTS_DIR / f"{name}.png"), img)
        print(f"  Input: {name}.png ({img.shape[1]}x{img.shape[0]})")

        if src["gt_src"].exists():
            gt = cv2.imread(str(src["gt_src"]), cv2.IMREAD_GRAYSCALE)
            if "crop" in src:
                y, x, h, w = src["crop"]
                gt = gt[y:y+h, x:x+w]
            cv2.imwrite(str(GT_DIR / f"{name}.png"), gt)


def run_pipeline_and_extract():
    """Run pipeline on inputs, extract masks to expected/."""
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging("default")

    # Create temp dir for pipeline output
    with tempfile.TemporaryDirectory() as tmp_out:
        params = BinarizerParams(
            path_imgs_dir=str(INPUTS_DIR),
            path_report_dir=tmp_out,
            path_logging_config="default",
            path_models_dir=str(PROJECT_ROOT / "pretrained_models"),
            model_name="unet_220805.pth",
            cache=True,
            cache_dir=str(Path(tmp_out) / "cache"),
            report_name="ref",
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

        print("\nRunning pipeline...")
        binarizer_pipeline(params)

        # Extract masks from pipeline output
        print("\nExtracting masks...")
        tmp_path = Path(tmp_out)
        for src in SOURCES:
            name = src["name"]
            img_filename = f"{name}.png"

            # Find masks dir for this image
            for report_dir in sorted(tmp_path.rglob("masks")):
                mask_files = list(report_dir.glob(f"{img_filename};*"))
                if not mask_files:
                    continue

                for mf in sorted(mask_files):
                    mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue

                    mf_name = mf.name
                    if "soft_bin" in mf_name:
                        out_name = f"{name}_soft.png"
                    elif "threshold_0.5" in mf_name:
                        out_name = f"{name}_t0.5.png"
                    elif "threshold_0.7" in mf_name:
                        out_name = f"{name}_t0.7.png"
                    else:
                        continue

                    cv2.imwrite(str(EXPECTED_DIR / out_name), mask)
                    print(f"  Expected: {out_name}")
                break


def main():
    print("=== Preparing inputs ===")
    prepare_inputs()

    print("\n=== Running pipeline & extracting reference masks ===")
    run_pipeline_and_extract()

    # Verify
    n_inputs = len(list(INPUTS_DIR.glob("*.png")))
    n_expected = len(list(EXPECTED_DIR.glob("*.png")))
    n_gt = len(list(GT_DIR.glob("*.png")))
    print(f"\nDone! inputs={n_inputs}, expected={n_expected}, gt={n_gt}")


if __name__ == "__main__":
    main()
