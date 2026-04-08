"""Level 2: Pipeline function contract tests.

These tests lock down the behavior of individual pipeline functions
on concrete inputs with reference values. If a library update changes
the output of any function, the corresponding test will fail.

Run: pytest tests/test_pipeline_contracts.py -v
"""
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")

DATA_DIR = Path(__file__).parent / "integration_data"
PROJECT_ROOT = Path(__file__).parent.parent


# ===========================================================================
# Group A: Preprocessing
# ===========================================================================

class TestPreprocessing:
    """Preprocessing functions must produce identical results."""

    def test_scale_factor_prep_reference(self):
        """cv2.resize 10x10 uint8 -> 20x20 (sf=2.0) matches reference."""
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        result = cv2.resize(img, (20, 20))
        ref = np.load(str(DATA_DIR / "scale_factor_prep_ref.npy"))
        np.testing.assert_array_equal(result, ref)

    def test_gaussian_blur_pipeline_reference(self):
        """GaussianBlur with kernel 5 on deterministic input matches reference."""
        row = np.arange(0, 256, 32, dtype=np.uint8)
        img = np.tile(row, (8, 1))
        result = cv2.GaussianBlur(img, (5, 5), 0)
        ref = np.load(str(DATA_DIR / "cv2_gaussian_blur_ref.npy"))
        np.testing.assert_array_equal(result, ref)

    def test_grayscale_load_dtype_range(self):
        """cv2.imread GRAYSCALE produces uint8 in [0, 255]."""
        img_path = DATA_DIR / "inputs" / "real_crop.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, "Failed to load %s" % img_path
        assert img.dtype == np.uint8
        assert img.min() >= 0
        assert img.max() <= 255
        assert img.shape == (128, 128)

    def test_to_tensor_then_smart_layers_shape(self):
        """Grayscale uint8 -> ToTensor -> make_img_with_smart_layers -> (3, H, W)."""
        from binarizers.legacy import make_img_with_smart_layers
        img = cv2.imread(str(DATA_DIR / "inputs" / "real_crop.png"), cv2.IMREAD_GRAYSCALE)
        tensor = T.ToTensor()(img)
        assert tensor.shape == (1, 128, 128)
        result = make_img_with_smart_layers(tensor)
        assert result.shape == (3, 128, 128)
        assert not torch.isnan(result).any()
        assert result.dtype == torch.float32


# ===========================================================================
# Group B: Model Inference
# ===========================================================================

class TestModelInference:
    """Model inference must be deterministic and produce expected results."""

    def test_unet_inference_deterministic(self, model_path):
        """Same input -> same output (two runs)."""
        from binarizers.legacy import UNet
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()

        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

    def test_seg_post_sigmoid_reference(self):
        """seg_post_m220805 sigmoid post-processing matches reference."""
        from binarizers.seg_model import seg_post_m220805

        ch0 = np.array([[0.5, 1.0, -0.5], [2.0, 0.0, -1.0]], dtype=np.float32)
        ch1 = np.array([[1.0, 0.5, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
        logits = torch.tensor(np.stack([ch0, ch1]))

        result = seg_post_m220805(logits)
        ref = np.load(str(DATA_DIR / "sigmoid_post_ref.npy"))
        np.testing.assert_allclose(result, ref, atol=1e-7)

    def test_seg_post_resize_reference(self):
        """Float32 mask resize (used in seg_model.py) matches reference."""
        row = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        mask = np.outer(row, row)
        result = cv2.resize(mask, (8, 8))
        ref = np.load(str(DATA_DIR / "seg_resize_ref.npy"))
        np.testing.assert_allclose(result, ref, atol=1e-6)


# ===========================================================================
# Group C: Mask Generation
# ===========================================================================

class TestMaskGeneration:
    """Mask generation pipeline steps must produce expected results."""

    def test_soft_mask_to_uint8(self):
        """Float [0,1] mask -> uint8 conversion matches expected values."""
        mask = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        # Pipeline does: (mask * 255).astype(np.uint8)
        result = (mask * 255).astype(np.uint8)
        expected = np.array([0, 63, 127, 191, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_mask_inversion_black_interest(self):
        """color_interest='black' inverts mask correctly."""
        mask = np.array([0.0, 0.3, 0.7, 1.0], dtype=np.float32)
        # Pipeline: (1 - mask) * 255 -> uint8
        inverted = ((1 - mask) * 255).astype(np.uint8)
        expected = np.array([255, 178, 76, 0], dtype=np.uint8)
        np.testing.assert_array_equal(inverted, expected)

    def test_threshold_mask_binary(self):
        """Thresholding produces only 0 and 255 (or 0 and 1 before scaling)."""
        mask = np.array([0.0, 0.3, 0.49, 0.5, 0.7, 1.0], dtype=np.float32)
        result = mask.copy()
        result[result < 0.5] = 0
        result[result >= 0.5] = 1
        unique = np.unique(result)
        assert set(unique) <= {0.0, 1.0}
        # After uint8 conversion
        result_u8 = (result * 255).astype(np.uint8)
        assert set(np.unique(result_u8)) <= {0, 255}

    def test_mask_save_load_roundtrip(self, tmp_path):
        """Grayscale uint8 mask: save as PNG -> load -> identical."""
        mask = np.array([
            [0, 50, 100, 150],
            [200, 255, 0, 128],
        ], dtype=np.uint8)
        path = str(tmp_path / "mask.png")
        cv2.imwrite(path, mask)
        loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(mask, loaded)


# ===========================================================================
# Group D: Report Functions
# ===========================================================================

class TestReportFunctions:
    """Report generation functions must produce valid output."""

    def test_plot_imgs_returns_valid_array(self):
        """plot_imgs returns a valid numpy array image."""
        from reports.plot import plot_imgs
        rng = np.random.RandomState(0)
        img1 = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        img2 = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        result = plot_imgs([img1, img2], ["A", "B"], dpi=72, fig_sz=3, img_cmaps=["gray"])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] > 0
        assert result.shape[1] > 0
        assert result.dtype == np.uint8
        assert result.mean() > 10  # not blank

    def test_plot_imgs_with_mask_returns_valid(self):
        """plot_imgs_with_mask returns a valid numpy array image."""
        from reports.plot import plot_imgs_with_mask
        rng = np.random.RandomState(0)
        img = rng.randint(0, 255, (64, 64), dtype=np.uint8)
        mask = rng.rand(64, 64).astype(np.float32)
        result = plot_imgs_with_mask(
            img, [mask], ["Original", "Mask"],
            dpi=72, fig_sz=3, show_img=True, masks_cmap=["spring"])
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.dtype == np.uint8
        assert result.mean() > 10

    def test_img2pdfimg_returns_image_object(self):
        """img2pdfimg converts numpy array to reportlab Image."""
        from reports.pdf_csv_report import img2pdfimg
        from reportlab.platypus import Image as RLImage
        img = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = img2pdfimg(img, downscale=1)
        assert isinstance(result, RLImage)

    def test_create_final_report_produces_pdf(self, tmp_path):
        """create_final_report creates a valid PDF file."""
        from reports.pdf_csv_report import create_final_report
        rng = np.random.RandomState(0)
        imgs = [rng.randint(0, 255, (100, 200, 3), dtype=np.uint8) for _ in range(2)]
        create_final_report(str(tmp_path), "test_report", imgs)
        pdf_path = tmp_path / "test_report.pdf"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 100
        header = pdf_path.read_bytes()[:5]
        assert header == b"%PDF-"
