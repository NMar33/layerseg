"""Tests for smart_contrast: fix current behavior before optimization.

Groups:
  A — Deterministic reference tests (manual calculations)
  B — Properties and invariants
  C — Edge cases
  D — Equivalence old vs new (added after optimization)
"""
import torch
import numpy as np
import pytest
from pathlib import Path

INTEGRATION_DIR = Path(__file__).parent / "integration_data"


# ---------------------------------------------------------------------------
# Group A: Deterministic reference tests
# ---------------------------------------------------------------------------

class TestReferenceManual:
    """Verify exact output on small known inputs (fixes off-by-one behavior)."""

    def test_reference_3x3_manual(self):
        """3x3 kernel on 4x4 image — hand-verified values.

        Off-by-one: fragment[:, h_pad+1, w_pad+1] = fragment[:, 2, 2]
        which is the BOTTOM-RIGHT corner of the 3x3 window, not center.
        """
        from binarizers.legacy import make_smart_contrast

        img = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8],
                              [0.2, 0.3, 0.4, 0.5],
                              [0.6, 0.7, 0.8, 0.9]]])

        result = make_smart_contrast(img, (3, 3))

        expected = torch.tensor([[[1.0, 1.0, 1.0, 0.0],
                                   [0.5, 0.5, 0.5, 0.0],
                                   [1.0, 1.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0]]])

        assert result.shape == expected.shape
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=0)

    def test_reference_3x3_spot_checks(self):
        """Verify individual pixel calculations by hand.

        [0,0]: padded[0:3,0:3] has center(2,2)=0.6, min=0, max=0.6 → 1.0
        [1,1]: padded[1:4,1:4] has center(2,2)=0.4, min=0.1, max=0.7 → 0.5
        [3,3]: padded[3:6,3:6] has center(2,2)=0(padding), min=0, max=0.9 → 0.0
        """
        from binarizers.legacy import make_smart_contrast

        img = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                              [0.5, 0.6, 0.7, 0.8],
                              [0.2, 0.3, 0.4, 0.5],
                              [0.6, 0.7, 0.8, 0.9]]])

        result = make_smart_contrast(img, (3, 3))

        assert abs(result[0, 0, 0].item() - 1.0) < 1e-6
        assert abs(result[0, 1, 1].item() - 0.5) < 1e-6
        assert abs(result[0, 3, 3].item() - 0.0) < 1e-6

    def test_reference_7x7_manual(self):
        """7x7 kernel on 8x8 linear gradient image.

        Image: arange(64)/63, so values go from 0.0 to 1.0.
        Off-by-one: center at fragment[:, 4, 4] instead of [:, 3, 3].
        """
        from binarizers.legacy import make_smart_contrast

        img = torch.arange(64).float().reshape(1, 8, 8) / 63.0
        result = make_smart_contrast(img, (7, 7))

        # Last column and last row are 0.0 (center falls into zero-padding)
        assert (result[0, :, 7] == 0.0).all(), "Last column should be 0 (padding)"
        assert (result[0, 7, :] == 0.0).all(), "Last row should be 0 (padding)"

        # Interior values should be in (0, 1] range
        interior = result[0, :6, :6]
        assert interior.min() > 0.0
        assert interior.max() <= 1.0

        # Specific spot checks
        assert abs(result[0, 0, 0].item() - 1/3) < 1e-4
        assert abs(result[0, 3, 3].item() - 2/3) < 1e-4

    def test_reference_full_pipeline(self):
        """make_img_with_smart_layers vs saved reference tensor."""
        ref_path = INTEGRATION_DIR / "smart_contrast_reference.pt"
        if not ref_path.exists():
            pytest.skip("Reference not generated. Run: python tests/integration_data/generate_smart_contrast_reference.py")

        from binarizers.legacy import make_img_with_smart_layers

        ref = torch.load(str(ref_path), map_location="cpu")
        result = make_img_with_smart_layers(ref["input"])

        torch.testing.assert_close(result, ref["output"], atol=1e-7, rtol=0)


# ---------------------------------------------------------------------------
# Group B: Properties and invariants
# ---------------------------------------------------------------------------

class TestInvariants:
    """Properties that must hold for any input."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_output_range(self, seed):
        """Output values in [0, 1] or NaN (before cleanup)."""
        from binarizers.legacy import make_smart_contrast

        torch.manual_seed(seed)
        img = torch.rand(1, 16, 16)
        result = make_smart_contrast(img, (3, 3))

        valid = result[~result.isnan()]
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0

    def test_nan_on_uniform_input(self):
        """Uniform image → interior pixels are NaN (0/0 division).

        Border pixels are NOT NaN because zero-padding introduces
        different values into the window.
        """
        from binarizers.legacy import make_smart_contrast

        img = torch.full((1, 10, 10), 0.5)
        result = make_smart_contrast(img, (3, 3))

        # Interior pixels (where window doesn't touch padding) should be NaN.
        # For 3x3 kernel (h_pad=1), with off-by-one, interior region where
        # the window is fully inside the original image starts later.
        # Check center pixel at least:
        assert torch.isnan(result[0, 1, 1]), "Interior pixel should be NaN"

        # Border row/col 0 should NOT be NaN (window touches zero-padding)
        assert not torch.isnan(result[0, 0, 0]), "Border pixel should not be NaN"

    def test_nan_cleanup_in_pipeline(self):
        """make_img_with_smart_layers output has no NaN."""
        from binarizers.legacy import make_img_with_smart_layers

        img = torch.full((1, 8, 8), 0.5)
        result = make_img_with_smart_layers(img)

        assert not torch.isnan(result).any()

    @pytest.mark.parametrize("h,w", [(8, 8), (16, 16), (10, 20), (32, 32)])
    def test_shape_preservation(self, h, w):
        """Output shape matches input for make_smart_contrast."""
        from binarizers.legacy import make_smart_contrast

        img = torch.rand(1, h, w)
        result = make_smart_contrast(img, (3, 3))
        assert result.shape == img.shape

    def test_channel_0_unchanged(self):
        """Channel 0 of make_img_with_smart_layers == original image."""
        from binarizers.legacy import make_img_with_smart_layers

        torch.manual_seed(99)
        img = torch.rand(1, 12, 12)
        result = make_img_with_smart_layers(img)

        assert result.shape[0] == 3
        torch.testing.assert_close(result[0], img[0])


# ---------------------------------------------------------------------------
# Group C: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary conditions and special inputs."""

    def test_single_bright_pixel(self):
        """One bright pixel on black background."""
        from binarizers.legacy import make_smart_contrast

        img = torch.zeros(1, 5, 5)
        img[0, 2, 2] = 1.0
        result = make_smart_contrast(img, (3, 3))

        # Due to off-by-one, the bright pixel at (2,2) appears as
        # center of patch at (1,1): fragment[2,2] = original[2,2] = 1.0
        assert result[0, 1, 1].item() == 1.0

        # All-zero patches → NaN
        assert torch.isnan(result[0, 0, 0])

        # Patches that see the bright pixel but center is 0 → result 0
        assert result[0, 1, 2].item() == 0.0

    def test_border_pixels_use_zero_padding(self):
        """Border pixels: window includes zero-padded region.

        For a 3x3 kernel, pixel [0,0] has window that includes
        the padding (zeros), which affects min/max calculation.
        """
        from binarizers.legacy import make_smart_contrast

        img = torch.full((1, 4, 4), 0.5)
        result = make_smart_contrast(img, (3, 3))

        # [0,0]: window includes padding zeros → min=0, max=0.5
        # center = fragment[2,2] = 0.5, result = (0.5-0)/(0.5-0) = 1.0
        assert result[0, 0, 0].item() == 1.0

        # [1,1]: fully interior, uniform → NaN
        assert torch.isnan(result[0, 1, 1])

    def test_minimum_size_3x3(self):
        """Minimum image size for 3x3 kernel."""
        from binarizers.legacy import make_smart_contrast

        img = torch.rand(1, 3, 3)
        result = make_smart_contrast(img, (3, 3))
        assert result.shape == (1, 3, 3)

    def test_minimum_size_7x7(self):
        """Minimum image size for 7x7 kernel."""
        from binarizers.legacy import make_smart_contrast

        img = torch.rand(1, 7, 7)
        result = make_smart_contrast(img, (7, 7))
        assert result.shape == (1, 7, 7)

    def test_non_square_image(self):
        """Rectangular image (H != W)."""
        from binarizers.legacy import make_smart_contrast

        img = torch.rand(1, 8, 16)
        result = make_smart_contrast(img, (3, 3))
        assert result.shape == (1, 8, 16)

        result7 = make_smart_contrast(img, (7, 7))
        assert result7.shape == (1, 8, 16)


# ---------------------------------------------------------------------------
# Group D: Equivalence — optimized vs original
# ---------------------------------------------------------------------------

class TestEquivalence:
    """Verify optimized (unfold) matches original (loop) implementation."""

    ATOL = 1e-7

    @pytest.mark.parametrize("seed,h,w", [
        (0, 16, 16), (1, 32, 32), (2, 10, 20), (3, 7, 13),
    ])
    def test_matches_original_random_3x3(self, seed, h, w):
        from binarizers.legacy import make_smart_contrast, _make_smart_contrast_original

        torch.manual_seed(seed)
        img = torch.rand(1, h, w)

        old = _make_smart_contrast_original(img, (3, 3))
        new = make_smart_contrast(img, (3, 3))

        # Both should have NaN in same positions
        assert torch.isnan(old).sum() == torch.isnan(new).sum()
        mask = ~torch.isnan(old)
        torch.testing.assert_close(new[mask], old[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.parametrize("seed,h,w", [
        (0, 16, 16), (1, 32, 32), (2, 10, 20),
    ])
    def test_matches_original_random_7x7(self, seed, h, w):
        from binarizers.legacy import make_smart_contrast, _make_smart_contrast_original

        torch.manual_seed(seed)
        img = torch.rand(1, h, w)

        old = _make_smart_contrast_original(img, (7, 7))
        new = make_smart_contrast(img, (7, 7))

        assert torch.isnan(old).sum() == torch.isnan(new).sum()
        mask = ~torch.isnan(old)
        torch.testing.assert_close(new[mask], old[mask], atol=self.ATOL, rtol=0)

    def test_matches_original_uniform(self):
        """Uniform image — both should produce NaN at same positions."""
        from binarizers.legacy import make_smart_contrast, _make_smart_contrast_original

        img = torch.full((1, 10, 10), 0.5)

        for conv in [(3, 3), (7, 7)]:
            old = _make_smart_contrast_original(img, conv)
            new = make_smart_contrast(img, conv)
            assert torch.isnan(old).sum() == torch.isnan(new).sum()
            mask = ~torch.isnan(old)
            torch.testing.assert_close(new[mask], old[mask], atol=self.ATOL, rtol=0)

    def test_matches_original_single_pixel(self):
        """Single bright pixel — both should match."""
        from binarizers.legacy import make_smart_contrast, _make_smart_contrast_original

        img = torch.zeros(1, 8, 8)
        img[0, 4, 4] = 1.0

        for conv in [(3, 3), (7, 7)]:
            old = _make_smart_contrast_original(img, conv)
            new = make_smart_contrast(img, conv)
            assert torch.isnan(old).sum() == torch.isnan(new).sum()
            mask = ~torch.isnan(old)
            torch.testing.assert_close(new[mask], old[mask], atol=self.ATOL, rtol=0)

    def test_matches_pipeline(self):
        """Full make_img_with_smart_layers: old vs new."""
        from binarizers.legacy import (
            make_img_with_smart_layers,
            _make_smart_contrast_original,
            make_smart_contrast,
        )

        torch.manual_seed(42)
        img = torch.rand(1, 16, 16)

        # Run pipeline with current (optimized) code
        result_new = make_img_with_smart_layers(img)

        # Run pipeline manually with original code
        ch1 = _make_smart_contrast_original(img, (3, 3))
        ch2 = _make_smart_contrast_original(1.0 - img, (7, 7))
        result_old = torch.cat([img, ch1, ch2], dim=0)
        result_old[result_old.isnan()] = 0

        torch.testing.assert_close(result_new, result_old, atol=self.ATOL, rtol=0)

    def test_matches_reference_tensor(self):
        """Optimized code vs saved reference .pt file."""
        ref_path = INTEGRATION_DIR / "smart_contrast_reference.pt"
        if not ref_path.exists():
            pytest.skip("Reference not generated")

        from binarizers.legacy import make_img_with_smart_layers

        ref = torch.load(str(ref_path), map_location="cpu")
        result = make_img_with_smart_layers(ref["input"])

        torch.testing.assert_close(result, ref["output"], atol=self.ATOL, rtol=0)

    def test_chunked_matches_non_chunked_small(self):
        """Forcing chunked processing on small image should match non-chunked."""
        from binarizers.legacy import make_smart_contrast

        torch.manual_seed(77)
        img = torch.rand(1, 32, 32)

        result_full = make_smart_contrast(img, (7, 7), max_elements=999_999_999)
        result_chunked = make_smart_contrast(img, (7, 7), max_elements=100)

        assert torch.isnan(result_full).sum() == torch.isnan(result_chunked).sum()
        mask = ~torch.isnan(result_full)
        torch.testing.assert_close(
            result_chunked[mask], result_full[mask], atol=self.ATOL, rtol=0
        )


# ---------------------------------------------------------------------------
# Group E: Chunked processing — large synthetic images
# ---------------------------------------------------------------------------

class TestChunkedLargeImages:
    """Verify chunked vs non-chunked on large images of various sizes.

    Uses max_elements to force different chunking strategies and verifies
    results are identical to non-chunked processing.
    """

    ATOL = 1e-7

    @staticmethod
    def _make_synthetic(h, w, seed=0, pattern="random"):
        """Create synthetic test images of various patterns."""
        torch.manual_seed(seed)
        if pattern == "random":
            return torch.rand(1, h, w)
        elif pattern == "gradient_h":
            row = torch.linspace(0, 1, h).unsqueeze(1).expand(1, h, w)
            return row
        elif pattern == "gradient_v":
            col = torch.linspace(0, 1, w).unsqueeze(0).expand(1, h, w)
            return col
        elif pattern == "checkerboard":
            r = torch.arange(h).unsqueeze(1).expand(h, w)
            c = torch.arange(w).unsqueeze(0).expand(h, w)
            return ((r + c) % 2).float().unsqueeze(0)
        elif pattern == "stripes":
            r = torch.arange(h).unsqueeze(1).expand(h, w)
            return ((r % 10) < 5).float().unsqueeze(0)
        else:
            raise ValueError(pattern)

    @pytest.mark.parametrize("size", [500, 1000, 2000, 3000])
    @pytest.mark.timeout(60)
    def test_chunked_random_3x3(self, size):
        """Random image, 3x3 kernel: chunked == non-chunked."""
        from binarizers.legacy import make_smart_contrast

        img = self._make_synthetic(size, size, seed=size, pattern="random")

        full = make_smart_contrast(img, (3, 3), max_elements=999_999_999)
        chunked = make_smart_contrast(img, (3, 3), max_elements=1_000_000)

        nan_full = torch.isnan(full)
        nan_chunked = torch.isnan(chunked)
        assert nan_full.sum() == nan_chunked.sum(), \
            "NaN count mismatch: full=%d, chunked=%d" % (nan_full.sum(), nan_chunked.sum())

        mask = ~nan_full
        torch.testing.assert_close(chunked[mask], full[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.parametrize("size", [500, 1000, 2000, 3000])
    @pytest.mark.timeout(60)
    def test_chunked_random_7x7(self, size):
        """Random image, 7x7 kernel: chunked == non-chunked."""
        from binarizers.legacy import make_smart_contrast

        img = self._make_synthetic(size, size, seed=size + 100, pattern="random")

        full = make_smart_contrast(img, (7, 7), max_elements=999_999_999)
        chunked = make_smart_contrast(img, (7, 7), max_elements=1_000_000)

        nan_full = torch.isnan(full)
        nan_chunked = torch.isnan(chunked)
        assert nan_full.sum() == nan_chunked.sum()

        mask = ~nan_full
        torch.testing.assert_close(chunked[mask], full[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.parametrize("pattern", [
        "gradient_h", "gradient_v", "checkerboard", "stripes",
    ])
    @pytest.mark.timeout(60)
    def test_chunked_patterns_1000(self, pattern):
        """Structured patterns at 1000x1000: chunked == non-chunked."""
        from binarizers.legacy import make_smart_contrast

        img = self._make_synthetic(1000, 1000, pattern=pattern)

        full = make_smart_contrast(img, (7, 7), max_elements=999_999_999)
        chunked = make_smart_contrast(img, (7, 7), max_elements=500_000)

        nan_full = torch.isnan(full)
        nan_chunked = torch.isnan(chunked)
        assert nan_full.sum() == nan_chunked.sum()

        mask = ~nan_full
        torch.testing.assert_close(chunked[mask], full[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.parametrize("max_elements", [
        100, 1_000, 10_000, 100_000, 1_000_000, 50_000_000,
    ])
    @pytest.mark.timeout(60)
    def test_various_chunk_sizes_500(self, max_elements):
        """Same image, different chunk sizes — all produce identical output."""
        from binarizers.legacy import make_smart_contrast

        img = self._make_synthetic(500, 500, seed=42, pattern="random")

        reference = make_smart_contrast(img, (7, 7), max_elements=999_999_999)
        result = make_smart_contrast(img, (7, 7), max_elements=max_elements)

        nan_ref = torch.isnan(reference)
        nan_res = torch.isnan(result)
        assert nan_ref.sum() == nan_res.sum()

        mask = ~nan_ref
        torch.testing.assert_close(result[mask], reference[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.timeout(60)
    def test_chunked_non_square_large(self):
        """Non-square large image: 500x2000."""
        from binarizers.legacy import make_smart_contrast

        img = self._make_synthetic(500, 2000, seed=7, pattern="random")

        full = make_smart_contrast(img, (7, 7), max_elements=999_999_999)
        chunked = make_smart_contrast(img, (7, 7), max_elements=500_000)

        nan_full = torch.isnan(full)
        nan_chunked = torch.isnan(chunked)
        assert nan_full.sum() == nan_chunked.sum()

        mask = ~nan_full
        torch.testing.assert_close(chunked[mask], full[mask], atol=self.ATOL, rtol=0)

    @pytest.mark.timeout(120)
    def test_chunked_full_pipeline_1000(self):
        """Full make_img_with_smart_layers at 1000x1000: chunked == non-chunked."""
        from binarizers.legacy import make_img_with_smart_layers

        img = self._make_synthetic(1000, 1000, seed=55, pattern="random")

        full = make_img_with_smart_layers(img, max_elements=999_999_999)
        chunked = make_img_with_smart_layers(img, max_elements=500_000)

        torch.testing.assert_close(chunked, full, atol=self.ATOL, rtol=0)
