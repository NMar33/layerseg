"""Level 4: Full pipeline golden master test.

Single comprehensive test that runs the entire pipeline end-to-end
on a reference input and verifies ALL outputs against golden data.

This is the ultimate regression test: if this passes after library
migration, the pipeline works correctly.

Run: pytest tests/test_pipeline_golden.py -v
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(__file__).parent / "integration_data"
GOLDEN_DIR = DATA_DIR / "golden_report"
INPUTS_DIR = DATA_DIR / "inputs"

GOLDEN_INPUT = "real_crop.png"


class TestPipelineGolden:
    """Golden master: full pipeline on reference input -> reference outputs."""

    @pytest.fixture(autouse=True)
    def run_pipeline(self, tmp_path):
        """Run the full pipeline and store results for assertions."""
        from entities import BinarizerParams
        from binarizer_pipeline import binarizer_pipeline
        from utils import setup_logging

        tmp_input = tmp_path / "input"
        tmp_input.mkdir()
        shutil.copy2(str(INPUTS_DIR / GOLDEN_INPUT), str(tmp_input / GOLDEN_INPUT))

        self.report_root = tmp_path / "reports"
        self.report_root.mkdir()

        setup_logging("default")

        params = BinarizerParams(
            path_imgs_dir=str(tmp_input),
            path_report_dir=str(self.report_root),
            path_logging_config="default",
            path_models_dir=str(PROJECT_ROOT / "pretrained_models"),
            model_name="unet_220805.pth",
            cache=True,
            cache_dir=str(tmp_path / "cache"),
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
            short_report_dir=str(tmp_path / "short"),
            device="cpu",
        )

        binarizer_pipeline(params)

        # Find output dirs
        report_dirs = [d for d in self.report_root.iterdir() if d.is_dir()]
        assert len(report_dirs) == 1
        self.report_dir = report_dirs[0]
        self.masks_dir = self.report_dir / "masks"

        # Extract masks
        self.masks = {}
        for mf in sorted(self.masks_dir.glob("*.png")):
            img = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
            name = mf.name
            if "soft_bin" in name:
                self.masks["soft"] = img
            elif "threshold_0.5" in name:
                self.masks["t05"] = img
            elif "threshold_0.7" in name:
                self.masks["t07"] = img

    def test_full_pipeline_golden(self):
        """Full pipeline produces correct outputs matching golden data."""
        # --- a) Soft mask matches golden (bit-exact for uint8) ---
        ref_soft = cv2.imread(
            str(GOLDEN_DIR / "soft_mask_uint8.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(
            self.masks["soft"], ref_soft,
            err_msg="Soft mask differs from golden reference")

        # --- b) Threshold masks bit-exact ---
        ref_t05 = cv2.imread(
            str(GOLDEN_DIR / "threshold_05.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(
            self.masks["t05"], ref_t05,
            err_msg="Threshold 0.5 mask differs from golden reference")

        ref_t07 = cv2.imread(
            str(GOLDEN_DIR / "threshold_07.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(
            self.masks["t07"], ref_t07,
            err_msg="Threshold 0.7 mask differs from golden reference")

        # --- c) Report structure ---
        assert self.masks_dir.exists(), "masks/ directory missing"
        assert len(list(self.masks_dir.glob("*.png"))) == 3, \
            "Expected 3 mask files"
        assert (self.report_dir / "01_preprocessed_imgs.png").exists(), \
            "Preprocessed image missing"

        # --- d) Visualizations: exist, valid shape, non-blank ---
        for f in self.report_dir.glob("*.png"):
            img = cv2.imread(str(f))
            assert img is not None, "Cannot read visualization %s" % f.name
            assert img.shape[0] > 0 and img.shape[1] > 0, \
                "Zero-size visualization %s" % f.name
            assert img.mean() > 10, \
                "Visualization %s appears blank" % f.name

        # --- e) PDF exists and is valid ---
        pdf_files = list(self.report_root.glob("*.pdf"))
        assert len(pdf_files) == 1, "Expected 1 PDF, got %d" % len(pdf_files)
        pdf = pdf_files[0]
        assert pdf.stat().st_size > 1000, \
            "PDF too small: %d bytes" % pdf.stat().st_size
        assert pdf.read_bytes()[:5] == b"%PDF-", "Invalid PDF header"
