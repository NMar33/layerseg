"""Level 3: Report regression tests.

These tests run the actual pipeline on golden input and verify:
- Mask data matches golden reference (exact or near-exact)
- Report directory structure is correct
- Visualizations are valid (structural checks, not pixel-exact)

Run: pytest tests/test_report_regression.py -v
"""
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import sys

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "integration_data"
GOLDEN_DIR = DATA_DIR / "golden_report"
INPUTS_DIR = DATA_DIR / "inputs"

GOLDEN_INPUT = "real_crop.png"


# ===========================================================================
# Fixture: run pipeline once, share results across tests
# ===========================================================================

@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    """Run the pipeline once on golden input and return paths."""
    from entities import BinarizerParams
    from binarizer_pipeline import binarizer_pipeline
    from utils import setup_logging

    tmp_root = tmp_path_factory.mktemp("golden_pipeline")
    tmp_input = tmp_root / "input"
    tmp_input.mkdir()
    shutil.copy2(str(INPUTS_DIR / GOLDEN_INPUT), str(tmp_input / GOLDEN_INPUT))

    tmp_report = tmp_root / "reports"
    tmp_report.mkdir()

    setup_logging("default")

    params = BinarizerParams(
        path_imgs_dir=str(tmp_input),
        path_report_dir=str(tmp_report),
        path_logging_config="default",
        path_models_dir=str(PROJECT_ROOT / "pretrained_models"),
        model_name="unet_220805.pth",
        cache=True,
        cache_dir=str(tmp_root / "cache"),
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
        short_report_dir=str(tmp_root / "short"),
        device="cpu",
    )

    binarizer_pipeline(params)

    # Find report dir
    report_dirs = [d for d in tmp_report.iterdir() if d.is_dir()]
    assert len(report_dirs) == 1
    report_dir = report_dirs[0]
    masks_dir = report_dir / "masks"

    # Extract masks
    masks = {}
    for mf in sorted(masks_dir.glob("*.png")):
        img = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        name = mf.name
        if "soft_bin" in name:
            masks["soft"] = img
        elif "threshold_0.5" in name:
            masks["t05"] = img
        elif "threshold_0.7" in name:
            masks["t07"] = img

    return {
        "report_dir": report_dir,
        "masks_dir": masks_dir,
        "report_root": tmp_report,
        "masks": masks,
    }


# ===========================================================================
# Group A: Mask Data Regression
# ===========================================================================

class TestMaskDataRegression:
    """Mask data must match golden reference exactly."""

    def test_golden_soft_mask_exact(self, pipeline_output):
        """Soft mask uint8 matches golden reference bit-exact."""
        result = pipeline_output["masks"]["soft"]
        ref = cv2.imread(str(GOLDEN_DIR / "soft_mask_uint8.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(result, ref)

    def test_golden_threshold_05_bitexact(self, pipeline_output):
        """Threshold mask t=0.5 matches golden reference bit-exact."""
        result = pipeline_output["masks"]["t05"]
        ref = cv2.imread(str(GOLDEN_DIR / "threshold_05.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(result, ref)

    def test_golden_threshold_07_bitexact(self, pipeline_output):
        """Threshold mask t=0.7 matches golden reference bit-exact."""
        result = pipeline_output["masks"]["t07"]
        ref = cv2.imread(str(GOLDEN_DIR / "threshold_07.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(result, ref)


# ===========================================================================
# Group B: Report Structure
# ===========================================================================

class TestReportStructure:
    """Report directory structure must match expected layout."""

    def test_report_directory_exists(self, pipeline_output):
        """Report directory was created."""
        assert pipeline_output["report_dir"].exists()
        assert pipeline_output["report_dir"].is_dir()

    def test_masks_directory_exists(self, pipeline_output):
        """masks/ subdirectory was created."""
        assert pipeline_output["masks_dir"].exists()
        assert pipeline_output["masks_dir"].is_dir()

    def test_preprocessed_image_exists(self, pipeline_output):
        """01_preprocessed_imgs.png was created."""
        f = pipeline_output["report_dir"] / "01_preprocessed_imgs.png"
        assert f.exists()
        assert f.stat().st_size > 0

    def test_mask_files_count(self, pipeline_output):
        """Expected number of mask files: 1 soft + 2 threshold = 3."""
        mask_files = list(pipeline_output["masks_dir"].glob("*.png"))
        assert len(mask_files) == 3

    def test_pdf_report_valid(self, pipeline_output):
        """PDF report exists, is non-trivial, and has valid header."""
        pdf_files = list(pipeline_output["report_root"].glob("*.pdf"))
        assert len(pdf_files) == 1
        pdf = pdf_files[0]
        assert pdf.stat().st_size > 1000
        header = pdf.read_bytes()[:5]
        assert header == b"%PDF-"

    def test_mask_filenames_convention(self, pipeline_output):
        """Mask filenames follow expected pattern."""
        mask_files = sorted([f.name for f in pipeline_output["masks_dir"].glob("*.png")])
        # Expected: {img_name};{desc}.png
        for name in mask_files:
            assert ";" in name, "Mask filename missing ';' separator: %s" % name
            assert name.startswith(GOLDEN_INPUT + ";"), \
                "Mask filename doesn't start with image name: %s" % name


# ===========================================================================
# Group C: Statistical Sanity
# ===========================================================================

class TestStatisticalSanity:
    """Visualization and mask outputs have expected statistical properties."""

    def test_visualization_not_blank(self, pipeline_output):
        """All PNG visualizations in report dir are not blank."""
        for f in pipeline_output["report_dir"].glob("*.png"):
            img = cv2.imread(str(f))
            assert img is not None, "Cannot read %s" % f
            mean_val = img.mean()
            assert mean_val > 10, "Image %s appears blank (mean=%.1f)" % (f.name, mean_val)
            assert img.min() < 245, "Image %s appears all white" % f.name

    def test_visualization_dimensions_stable(self, pipeline_output):
        """Visualization PNG dimensions are within ±5%% of golden."""
        manifest = json.loads((GOLDEN_DIR / "manifest.json").read_text())
        golden_files = manifest["golden_files"]

        for f in pipeline_output["report_dir"].glob("*.png"):
            img = cv2.imread(str(f))
            # Find corresponding golden file
            golden_key = None
            for gk in golden_files:
                if gk.startswith("viz_") and f.name in gk:
                    golden_key = gk
                    break
            if golden_key is None:
                continue  # no golden match, skip

            golden_img = cv2.imread(str(GOLDEN_DIR / golden_key))
            if golden_img is None:
                continue

            h_ratio = img.shape[0] / golden_img.shape[0]
            w_ratio = img.shape[1] / golden_img.shape[1]
            assert 0.95 <= h_ratio <= 1.05, \
                "Height changed by %.1f%% for %s" % ((h_ratio - 1) * 100, f.name)
            assert 0.95 <= w_ratio <= 1.05, \
                "Width changed by %.1f%% for %s" % ((w_ratio - 1) * 100, f.name)

    def test_soft_mask_statistics(self, pipeline_output):
        """Soft mask statistics are within expected range."""
        mask = pipeline_output["masks"]["soft"]
        golden = cv2.imread(str(GOLDEN_DIR / "soft_mask_uint8.png"), cv2.IMREAD_GRAYSCALE)

        # Mean within ±10%
        golden_mean = golden.mean()
        mask_mean = mask.mean()
        if golden_mean > 0:
            ratio = mask_mean / golden_mean
            assert 0.9 <= ratio <= 1.1, \
                "Soft mask mean changed: golden=%.1f, current=%.1f" % (golden_mean, mask_mean)

        # Range check
        assert mask.min() >= 0
        assert mask.max() <= 255
        # Not degenerate (all same value)
        assert mask.std() > 1.0, "Soft mask has near-zero variance"
