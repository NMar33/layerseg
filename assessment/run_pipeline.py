"""Entry point: run binarizer pipeline on synthetic assessment images.

Usage:
    python -m assessment.run_pipeline
    python -m assessment.run_pipeline --config assessment/configs/pipeline_config.yaml
"""
import argparse
from pathlib import Path

import yaml

from .pipeline_runner.runner import run_on_category
from .metadata.io import save_metadata, build_pipeline_metadata


def main(config_path="assessment/configs/pipeline_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    project_root = config.get("project_root", ".")
    input_base = Path(config["input_dir"])
    output_base = Path(config["output_dir"])
    presets = config.get("presets", {})

    all_results = []

    categories = sorted([
        d.name for d in input_base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(categories)} categories: {categories}")
    print(f"Running {len(presets)} presets: {list(presets.keys())}")
    print()

    for preset_name, preset_config in presets.items():
        print(f"=== Preset: {preset_name} ===")
        for cat_name in categories:
            cat_input = input_base / cat_name
            cat_output = output_base / preset_name / cat_name

            n_imgs = sum(len(list(cat_input.glob(ext))) for ext in ["*.png", "*.jpg", "*.tif"])
            print(f"  {cat_name}: {n_imgs} images ...", flush=True)

            results = run_on_category(
                project_root=project_root,
                input_dir=cat_input,
                output_dir=cat_output,
                preset_name=preset_name,
                preset_config=preset_config,
                pipeline_config=config,
            )
            all_results.extend(results)
            total_time = sum(r["processing_time_seconds"] for r in results)
            print(f"    done ({total_time:.1f}s, {len(results)} scale groups)")

        print()

    metadata = build_pipeline_metadata(config_path, all_results)
    save_metadata(metadata, config["metadata_path"])
    print(f"Pipeline metadata: {config['metadata_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run binarizer on assessment images")
    parser.add_argument("--config", default="assessment/configs/pipeline_config.yaml")
    args = parser.parse_args()
    main(args.config)
