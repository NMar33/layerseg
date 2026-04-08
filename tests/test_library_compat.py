"""Level 1: Library compatibility smoke tests.

These tests verify that basic library operations produce the same results
as before migration. Each test uses deterministic inputs and compares
against pre-generated reference data.

Run: pytest tests/test_library_compat.py -v
"""
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "integration_data"


# ---------------------------------------------------------------------------
# Helpers: deterministic input arrays (must match generate_library_references.py)
# ---------------------------------------------------------------------------

def _make_uint8_4x4():
    return np.array([
        [10, 50, 100, 200],
        [20, 60, 110, 210],
        [30, 70, 120, 220],
        [40, 80, 130, 230],
    ], dtype=np.uint8)


def _make_float32_4x4():
    return np.array([
        [0.1, 0.25, 0.5, 0.75],
        [0.15, 0.3, 0.55, 0.8],
        [0.2, 0.35, 0.6, 0.85],
        [0.0, 0.4, 0.65, 1.0],
    ], dtype=np.float32)


def _make_uint8_8x8():
    row = np.arange(0, 256, 32, dtype=np.uint8)
    return np.tile(row, (8, 1))


# ---------------------------------------------------------------------------
# cv2 tests
# ---------------------------------------------------------------------------

class TestCv2Compat:
    """OpenCV operations must produce identical results across versions."""

    def test_cv2_resize_uint8_reference(self):
        """cv2.resize on uint8 4x4 -> 8x8 matches reference."""
        img = _make_uint8_4x4()
        result = cv2.resize(img, (8, 8))
        ref = np.load(str(DATA_DIR / "cv2_resize_uint8_ref.npy"))
        np.testing.assert_array_equal(result, ref)

    def test_cv2_resize_float32_reference(self):
        """cv2.resize on float32 4x4 -> 8x8 matches reference."""
        img = _make_float32_4x4()
        result = cv2.resize(img, (8, 8))
        ref = np.load(str(DATA_DIR / "cv2_resize_float32_ref.npy"))
        np.testing.assert_allclose(result, ref, atol=1e-6)

    def test_cv2_gaussian_blur_reference(self):
        """cv2.GaussianBlur on 8x8 with kernel (5,5) matches reference.

        Single-threaded to avoid OpenCV multithreading race on small images.
        """
        img = _make_uint8_8x8()
        orig_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            result = cv2.GaussianBlur(img, (5, 5), 0)
        finally:
            cv2.setNumThreads(orig_threads)
        ref = np.load(str(DATA_DIR / "cv2_gaussian_blur_ref.npy"))
        np.testing.assert_array_equal(result, ref)

    def test_cv2_imwrite_imread_roundtrip(self, tmp_path):
        """Write grayscale PNG then read back -> identical."""
        img = _make_uint8_4x4()
        path = str(tmp_path / "test.png")
        cv2.imwrite(path, img)
        loaded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(img, loaded)


# ---------------------------------------------------------------------------
# torch + torchvision tests
# ---------------------------------------------------------------------------

class TestTorchCompat:
    """PyTorch/torchvision operations must produce identical results."""

    def test_torch_load_state_dict(self, model_path):
        """torch.load can load the .pth model file."""
        from binarizers.legacy import UNet
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        state = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        # Verify model produces output of correct shape
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)
            out = model(x)
            assert out.shape == (1, 2, 32, 32)

    def test_torchvision_to_tensor_values(self):
        """ToTensor() converts uint8 numpy -> float32 tensor with correct values."""
        import torchvision.transforms as T
        img = _make_uint8_4x4()
        result = T.ToTensor()(img)
        ref = torch.load(str(DATA_DIR / "to_tensor_ref.pt"))
        torch.testing.assert_close(result, ref, atol=0, rtol=0)

    def test_torch_fpad_reference(self):
        """F.pad with zero padding matches reference."""
        inp = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = F.pad(inp.unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0).squeeze(0)
        ref = torch.load(str(DATA_DIR / "fpad_ref.pt"))
        torch.testing.assert_close(result, ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# numpy tests
# ---------------------------------------------------------------------------

class TestNumpyCompat:
    """NumPy operations used in the pipeline must behave identically."""

    def test_numpy_uint8_cast_and_round(self):
        """float -> round -> astype(uint8) produces expected values."""
        vals = np.array([0.4, 0.5, 0.6, 1.0, 254.7, 255.0], dtype=np.float64)
        result = np.round(vals).astype(np.uint8)
        expected = np.array([0, 0, 1, 1, 255, 255], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_numpy_threshold_operations(self):
        """Threshold logic used in report_generator matches expected."""
        mask = np.array([0.0, 0.3, 0.49, 0.5, 0.7, 1.0], dtype=np.float32)
        result = mask.copy()
        result[result < 0.5] = 0
        result[result >= 0.5] = 1
        expected = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# PIL + matplotlib + yaml/marshmallow + reportlab tests
# ---------------------------------------------------------------------------

class TestPilCompat:
    """PIL Image <-> numpy roundtrip must be lossless."""

    def test_pil_numpy_roundtrip(self):
        """numpy -> PIL.Image -> BytesIO -> PIL.Image -> numpy is identical."""
        arr = _make_uint8_8x8()
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        loaded = Image.open(buf)
        result = np.array(loaded)
        np.testing.assert_array_equal(arr, result)


class TestMatplotlibCompat:
    """Matplotlib figure rendering produces valid output."""

    def test_matplotlib_figure_to_array(self):
        """Simple figure -> savefig -> PIL -> numpy produces valid array."""
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=72)
        ax.imshow(np.eye(10), cmap="gray")
        ax.set_title("test")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        arr = np.array(img)
        # Valid image: has height, width, channels; not blank
        assert arr.ndim == 3
        assert arr.shape[0] > 0
        assert arr.shape[1] > 0
        assert arr.dtype == np.uint8
        assert arr.mean() > 10  # not all black
        assert arr.min() < 245  # not all white


class TestYamlMarshmallowCompat:
    """YAML parsing + marshmallow schema validation works correctly."""

    def test_yaml_marshmallow_config_load(self, config_path):
        """Load real config.yaml through marshmallow schema."""
        from entities import read_binarizer_params, BinarizerParams
        params = read_binarizer_params(str(config_path))
        assert isinstance(params, BinarizerParams)
        assert isinstance(params.scale_factors, list)
        assert all(isinstance(sf, float) for sf in params.scale_factors)
        assert isinstance(params.device, str)
        assert isinstance(params.cache, bool)
        assert isinstance(params.report_dpi, int)


class TestReportlabCompat:
    """ReportLab PDF generation works correctly."""

    def test_reportlab_minimal_pdf(self, tmp_path):
        """Create a minimal PDF and verify it's valid."""
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4

        pdf_path = str(tmp_path / "test.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("Test paragraph", styles["Normal"])]
        doc.build(story)

        # PDF exists and is valid
        pdf_file = Path(pdf_path)
        assert pdf_file.exists()
        assert pdf_file.stat().st_size > 100
        header = pdf_file.read_bytes()[:5]
        assert header == b"%PDF-"
