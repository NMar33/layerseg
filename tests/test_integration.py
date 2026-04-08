"""Integration tests: compare pipeline output with saved reference masks.

These tests run the full binarizer pipeline on 5 small (128x128) images
and verify the output matches previously saved reference data.

If a test fails after code changes, either:
- The change broke something (fix it)
- The change intentionally altered behavior (re-run generate_reference.py)

Run: pytest tests/test_integration.py -v
Time: ~30-60 seconds (5 images × 128x128 × scale=1)
"""
import sys
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# Image names used in tests
IMAGE_NAMES = [
    "real_crop",
    "synth_type02_crop",
    "wavy_thin_gray_bg_128x128",
    "horiz_uniform_128x128",
    "checker_uniform_128x128",
]

THRESHOLDS = [0.5, 0.7]

INTEGRATION_DIR = Path(__file__).parent / "integration_data"
INPUTS_DIR = INTEGRATION_DIR / "inputs"
EXPECTED_DIR = INTEGRATION_DIR / "expected"
GT_DIR = INTEGRATION_DIR / "ground_truth"


def _check_reference_exists():
    """Check that reference data has been generated."""
    if not INPUTS_DIR.exists() or len(list(INPUTS_DIR.glob("*.png"))) == 0:
        pytest.skip("Reference data not generated. Run: python tests/integration_data/generate_reference.py")


def _run_pipeline_once():
    """Run pipeline on all test inputs, return dict of output masks."""
    from entities import BinarizerParams
    from binarizer_pipeline import binarizer_pipeline
    from utils import setup_logging

    setup_logging("default")

    project_root = Path(__file__).parent.parent
    tmp_out = tempfile.mkdtemp(prefix="integration_test_")

    params = BinarizerParams(
        path_imgs_dir=str(INPUTS_DIR),
        path_report_dir=tmp_out,
        path_logging_config="default",
        path_models_dir=str(project_root / "pretrained_models"),
        model_name="unet_220805.pth",
        cache=True,
        cache_dir=str(Path(tmp_out) / "cache"),
        report_name="inttest",
        scale_factors=[1],
        gaussian_blur=False,
        gaussian_blur_kernel_size=5,
        binarizer_thresholds=THRESHOLDS,
        original_img_color_map="gray",
        imgs_in_row=3,
        color_interest="black",
        report_dpi=100,
        report_fig_sz=3,
        report_short=False,
        short_report_dir=str(Path(tmp_out) / "short"),
        device="cpu",
    )

    binarizer_pipeline(params)

    # Extract masks
    results = {}
    tmp_path = Path(tmp_out)
    for img_name in IMAGE_NAMES:
        img_filename = f"{img_name}.png"
        results[img_name] = {}

        for masks_dir in sorted(tmp_path.rglob("masks")):
            mask_files = list(masks_dir.glob(f"{img_filename};*"))
            if not mask_files:
                continue
            for mf in mask_files:
                mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                if "soft_bin" in mf.name:
                    results[img_name]["soft"] = mask
                elif "threshold_0.5" in mf.name:
                    results[img_name]["t0.5"] = mask
                elif "threshold_0.7" in mf.name:
                    results[img_name]["t0.7"] = mask
            break

    # Cleanup
    shutil.rmtree(tmp_out, ignore_errors=True)
    return results


# Module-level cache so pipeline runs only once per test session
_pipeline_cache = None


def get_pipeline_outputs():
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = _run_pipeline_once()
    return _pipeline_cache


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestSoftMaskMatchesReference:
    """Soft mask should be near-identical to saved reference."""

    @pytest.fixture(autouse=True)
    def check_data(self):
        _check_reference_exists()

    @pytest.mark.parametrize("img_name", IMAGE_NAMES)
    def test_soft_mask(self, img_name):
        outputs = get_pipeline_outputs()
        actual = outputs[img_name].get("soft")
        assert actual is not None, f"No soft mask produced for {img_name}"

        ref_path = EXPECTED_DIR / f"{img_name}_soft.png"
        assert ref_path.exists(), f"Reference not found: {ref_path}"
        expected = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

        assert actual.shape == expected.shape, \
            f"Shape mismatch: {actual.shape} vs {expected.shape}"

        # Allow max 1 pixel value difference (float rounding)
        max_diff = int(np.max(np.abs(actual.astype(int) - expected.astype(int))))
        assert max_diff <= 1, \
            f"Soft mask differs by up to {max_diff} pixel values (max allowed: 1)"


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestThresholdMaskMatchesReference:
    """Binary masks at specific thresholds should be identical to reference."""

    @pytest.fixture(autouse=True)
    def check_data(self):
        _check_reference_exists()

    @pytest.mark.parametrize("img_name", IMAGE_NAMES)
    @pytest.mark.parametrize("threshold", THRESHOLDS)
    def test_threshold_mask(self, img_name, threshold):
        outputs = get_pipeline_outputs()
        key = f"t{threshold}"
        actual = outputs[img_name].get(key)
        assert actual is not None, f"No {key} mask produced for {img_name}"

        ref_path = EXPECTED_DIR / f"{img_name}_{key}.png"
        assert ref_path.exists(), f"Reference not found: {ref_path}"
        expected = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)

        assert actual.shape == expected.shape
        assert np.array_equal(actual, expected), \
            f"Threshold mask {key} differs. Mismatched pixels: {np.sum(actual != expected)}"


@pytest.mark.integration
@pytest.mark.timeout(120)
class TestIoUAboveMinimum:
    """IoU with ground truth should be above a minimum for each image."""

    # Minimum IoU per image (from current assessment results, with margin)
    MIN_IOU = {
        "real_crop": 0.05,
        "synth_type02_crop": 0.05,
        "wavy_thin_gray_bg_128x128": 0.05,
        "horiz_uniform_128x128": 0.05,
        "checker_uniform_128x128": 0.05,
    }

    @pytest.fixture(autouse=True)
    def check_data(self):
        _check_reference_exists()

    @pytest.mark.parametrize("img_name", IMAGE_NAMES)
    def test_iou(self, img_name):
        outputs = get_pipeline_outputs()
        # Use t=0.5 mask (more permissive threshold, better recall)
        actual = outputs[img_name].get("t0.5")
        assert actual is not None

        gt_path = GT_DIR / f"{img_name}.png"
        if not gt_path.exists():
            pytest.skip(f"No GT for {img_name}")
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

        # Pipeline mask: lines=BLACK(0), GT: lines=WHITE(255)
        pred_fg = (actual <= 127).astype(bool)
        gt_fg = (gt > 127).astype(bool)

        intersection = np.sum(pred_fg & gt_fg)
        union = np.sum(pred_fg | gt_fg)
        iou = intersection / union if union > 0 else 1.0

        min_iou = self.MIN_IOU.get(img_name, 0.05)
        assert iou >= min_iou, \
            f"IoU {iou:.4f} below minimum {min_iou} for {img_name}"
