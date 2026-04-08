"""End-to-end test: load image, preprocess, segment, verify output."""
import pytest
import torch
import joblib
import numpy as np
import cv2


@pytest.mark.timeout(120)
class TestEndToEnd:
    """Full pipeline test on a single small image crop."""

    def test_single_image_segmentation(self, model_path, example_imgs_dir):
        """Loads a real image, preprocesses, runs inference, checks output."""
        from binarizers.legacy import make_img_with_smart_layers
        from binarizers.seg_model import seg_post_m220805
        from torchvision import transforms as T

        # Load model
        from binarizers.legacy import UNet
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        model.load_state_dict(torch.load(model_path.as_posix(), map_location="cpu"))
        model.eval()

        # Load a real image (take a small crop for speed)
        img_files = [f for f in example_imgs_dir.iterdir() if f.name != ".gitkeep"]
        assert len(img_files) > 0
        img_gray = cv2.imread(img_files[0].as_posix(), cv2.IMREAD_GRAYSCALE)
        assert img_gray is not None

        # Crop to small size for fast test
        h, w = img_gray.shape
        crop = img_gray[:min(h, 64), :min(w, 64)]

        # ToTensor + smart layers
        transform = T.ToTensor()
        img_tensor = transform(crop)  # [1, H, W]
        img_3ch = make_img_with_smart_layers(img_tensor)  # [3, H, W]
        assert img_3ch.shape[0] == 3

        # Inference
        with torch.no_grad():
            logits = model(img_3ch.unsqueeze(0))  # [1, 2, H, W]
        assert logits.shape[0] == 1
        assert logits.shape[1] == 2

        # Post-processing
        mask = seg_post_m220805(logits.squeeze())
        assert mask.shape == crop.shape
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

        # Threshold
        binary_mask = (mask >= 0.7).astype(np.uint8)
        assert set(np.unique(binary_mask)).issubset({0, 1})
