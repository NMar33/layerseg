"""Test smart layer preprocessing."""
import torch
import numpy as np
import pytest


class TestSmartLayers:
    """Tests for smart contrast layer creation."""

    def test_make_padding(self):
        from binarizers.legacy import make_padding
        img = torch.ones(1, 10, 10)
        padded = make_padding(img, (2, 3))
        assert padded.shape == (1, 14, 16)
        # Center should be 1
        assert padded[0, 2, 3] == 1.0
        # Padding should be 0
        assert padded[0, 0, 0] == 0.0

    def test_make_smart_contrast_shape(self):
        from binarizers.legacy import make_smart_contrast
        img = torch.rand(1, 20, 20)
        result = make_smart_contrast(img, (3, 3))
        assert result.shape == img.shape

    def test_make_smart_contrast_range(self):
        """Output should be in [0, 1] (plus NaN for uniform regions -> 0 after cleanup)."""
        from binarizers.legacy import make_smart_contrast
        img = torch.rand(1, 16, 16)
        result = make_smart_contrast(img, (3, 3))
        result[result.isnan()] = 0
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.timeout(30)
    def test_make_img_with_smart_layers(self):
        """Full smart layer creation: 1ch -> 3ch. Uses small image for speed."""
        from binarizers.legacy import make_img_with_smart_layers
        img = torch.rand(1, 16, 16)
        result = make_img_with_smart_layers(img)
        assert result.shape == (3, 16, 16)
        # No NaN in final output
        assert not torch.isnan(result).any()

    @pytest.mark.timeout(30)
    def test_smart_layers_deterministic(self):
        """Same input should produce same smart layers."""
        from binarizers.legacy import make_img_with_smart_layers
        img = torch.rand(1, 12, 12)
        r1 = make_img_with_smart_layers(img)
        r2 = make_img_with_smart_layers(img)
        assert torch.equal(r1, r2)
