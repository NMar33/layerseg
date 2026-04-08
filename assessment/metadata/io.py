"""Read/write JSON metadata files."""
import json
from datetime import datetime
from pathlib import Path


def save_metadata(data, path):
    """Save metadata dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_metadata(path):
    """Load metadata dict from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_generation_metadata(config_path, seed, images_info):
    """Build generation metadata dict.

    Args:
        config_path: path to generate_config.yaml used
        seed: random seed
        images_info: list of dicts with keys:
            filename, gt_filename, category, variation, priority, size, params
    """
    return {
        "generated_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "seed": seed,
        "images": images_info,
    }


def build_pipeline_metadata(config_path, results_info):
    """Build pipeline run metadata dict.

    Args:
        config_path: path to pipeline_config.yaml used
        results_info: list of dicts with keys:
            preset, input_image, output_dir, settings, processing_time_seconds
    """
    return {
        "run_at": datetime.now().isoformat(),
        "config_path": str(config_path),
        "results": results_info,
    }
