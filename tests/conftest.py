"""Pytest configuration for Bin4_PhasIAn tests.

Handles the project's import structure: all source imports are bare
(e.g., `from entities import ...`), so we add `src/` to sys.path.
"""
import sys
from pathlib import Path
import pytest

# Project root (where configs/, data/, pretrained_models/ live)
PROJECT_ROOT = Path(__file__).parent.parent

# Add src/ to path so bare imports work (from entities import ..., etc.)
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def model_path():
    return PROJECT_ROOT / "pretrained_models" / "unet_220805.pth"


@pytest.fixture
def example_imgs_dir():
    return PROJECT_ROOT / "data" / "example_imgs"


@pytest.fixture
def config_path():
    return PROJECT_ROOT / "configs" / "config.yaml"
