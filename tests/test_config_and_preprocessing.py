"""Test config loading and image preprocessing."""
import pytest
import numpy as np
import cv2


class TestConfig:
    """Tests for YAML config loading and validation."""

    def test_config_file_exists(self, config_path):
        assert config_path.exists()

    def test_config_loads(self, config_path):
        from entities import read_binarizer_params
        params = read_binarizer_params(config_path.as_posix())
        assert params is not None

    def test_config_fields(self, config_path):
        from entities import read_binarizer_params
        params = read_binarizer_params(config_path.as_posix())
        assert params.model_name.startswith("unet_220805")
        assert params.path_models_dir == "pretrained_models"
        assert isinstance(params.scale_factors, list)
        assert len(params.scale_factors) > 0
        assert isinstance(params.binarizer_thresholds, list)
        assert len(params.binarizer_thresholds) > 0
        assert params.color_interest in ("black", "white")
        assert params.device in ("cpu", "cuda")

    def test_colab_config_loads(self, project_root):
        from entities import read_binarizer_params
        colab_cfg = project_root / "configs" / "config_colab.yaml"
        if colab_cfg.exists():
            params = read_binarizer_params(colab_cfg.as_posix())
            assert params is not None


class TestPreprocessing:
    """Tests for image loading and preprocessing."""

    def test_example_images_exist(self, example_imgs_dir):
        imgs = list(example_imgs_dir.glob("*"))
        real_imgs = [f for f in imgs if f.name != ".gitkeep"]
        assert len(real_imgs) >= 1, "No example images found"

    def test_image_loads_rgb(self, example_imgs_dir):
        img_path = list(example_imgs_dir.glob("*"))[0]
        if img_path.name == ".gitkeep":
            img_path = list(example_imgs_dir.glob("*"))[1]
        img = cv2.imread(img_path.as_posix())
        assert img is not None, f"Failed to load {img_path}"
        assert len(img.shape) == 3
        assert img.shape[2] == 3

    def test_image_loads_grayscale(self, example_imgs_dir):
        img_path = list(example_imgs_dir.glob("*"))[0]
        if img_path.name == ".gitkeep":
            img_path = list(example_imgs_dir.glob("*"))[1]
        img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert len(img.shape) == 2

    def test_scale_factor_prep(self):
        from binarizers.load_preprocess import scale_factor_prep
        img = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
        imgs, descs_s, descs_f = scale_factor_prep(img, [2.0, 1.0, 0.5], "test_", "Test ")
        assert len(imgs) == 3
        assert imgs[0].shape == (200, 400)  # 2x
        assert imgs[1].shape == (100, 200)  # 1x
        assert imgs[2].shape == (50, 100)   # 0.5x
