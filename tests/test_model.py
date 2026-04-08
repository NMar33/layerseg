"""Test model loading and inference."""
import json
import hashlib
from pathlib import Path

import pytest
import torch
import joblib
import numpy as np

INTEGRATION_DIR = Path(__file__).parent / "integration_data"


def _load_model(model_path):
    """Load model from .pth (state_dict) or .model (joblib) format."""
    from binarizers.legacy import UNet
    path_str = model_path.as_posix()
    if path_str.endswith(".pth"):
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        model.load_state_dict(torch.load(path_str, map_location="cpu"))
    else:
        model = joblib.load(path_str)
    return model


class TestModelLoad:
    """Tests for loading the pre-trained model."""

    def test_model_file_exists(self, model_path):
        assert model_path.exists(), f"Model file not found: {model_path}"

    def test_model_loads(self, model_path):
        model = _load_model(model_path)
        assert model is not None

    def test_model_is_unet(self, model_path):
        from binarizers.legacy import UNet
        model = _load_model(model_path)
        assert isinstance(model, UNet)

    def test_model_channels(self, model_path):
        model = _load_model(model_path)
        assert model.n_channels == 3
        assert model.n_classes == 2


class TestModelInference:
    """Tests for model forward pass."""

    @pytest.fixture
    def model(self, model_path):
        model = _load_model(model_path)
        model.eval()
        return model

    def test_forward_pass_shape(self, model):
        """Model should accept [B, 3, H, W] and return [B, 2, H, W]."""
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == torch.Size([1, 2, 64, 64])

    def test_forward_pass_various_sizes(self, model):
        """Model should handle different spatial dimensions (multiples of 16)."""
        for size in [(128, 128), (64, 96), (48, 64)]:
            x = torch.randn(1, 3, *size)
            with torch.no_grad():
                y = model(x)
            assert y.shape == torch.Size([1, 2, *size]), f"Failed for input size {size}"

    def test_output_deterministic(self, model):
        """Same input should produce same output."""
        torch.manual_seed(42)
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            y1 = model(x).numpy()
            y2 = model(x).numpy()
        np.testing.assert_array_equal(y1, y2)

    def test_postprocessing_sigmoid(self, model):
        """Post-processing should produce values in [0, 1]."""
        from binarizers.seg_model import seg_post_m220805
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            logits = model(x).squeeze()
        mask = seg_post_m220805(logits)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
        assert mask.shape == (64, 64)


class TestModelWeights:
    """Tests that verify model weights and structure haven't changed."""

    @pytest.fixture
    def model(self, model_path):
        return _load_model(model_path)

    @pytest.fixture
    def weight_info(self):
        path = INTEGRATION_DIR / "model_weight_info.json"
        if not path.exists():
            pytest.skip("Reference not generated. Run: python tests/integration_data/generate_model_reference.py")
        with open(str(path)) as f:
            return json.load(f)

    def test_state_dict_structure(self, model, weight_info):
        """State dict should have the same keys and shapes as reference."""
        state = model.state_dict()
        assert set(state.keys()) == set(weight_info.keys()), \
            f"Key mismatch: {set(state.keys()) ^ set(weight_info.keys())}"
        for key in state:
            assert list(state[key].shape) == weight_info[key]["shape"], \
                f"Shape mismatch for {key}: {list(state[key].shape)} vs {weight_info[key]['shape']}"

    def test_total_parameters(self, model):
        """Total parameter count should match known value."""
        total = sum(p.numel() for p in model.state_dict().values())
        assert total == 31_025_922

    def test_weight_checksums(self, model, weight_info):
        """MD5 checksum of each weight tensor should match reference."""
        state = model.state_dict()
        mismatches = []
        for key, tensor in state.items():
            md5 = hashlib.md5(tensor.numpy().tobytes()).hexdigest()
            if md5 != weight_info[key]["md5"]:
                mismatches.append(key)
        assert len(mismatches) == 0, \
            f"Weight checksums differ for {len(mismatches)} layers: {mismatches}"

    def test_reference_output(self, model):
        """Output on fixed input should match saved reference exactly."""
        ref_path = INTEGRATION_DIR / "model_reference_output.pt"
        if not ref_path.exists():
            pytest.skip("Reference not generated.")
        ref = torch.load(str(ref_path))

        model.eval()
        with torch.no_grad():
            y = model(ref["input"])

        torch.testing.assert_close(y, ref["output"], atol=1e-5, rtol=1e-5,
            msg=f"Output differs. Max diff: {(y - ref['output']).abs().max().item()}")
