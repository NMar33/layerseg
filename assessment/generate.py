"""Entry point: generate synthetic images for assessment.

Usage:
    python -m assessment.generate
    python -m assessment.generate --config assessment/configs/generate_config.yaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from .generators import (
    WavyLinesGenerator, StraightLinesGenerator,
    CheckerboardGenerator, CirclesEllipsesGenerator, CompositeGenerator,
)
from .generators.base import get_all_generators
from .metadata.io import save_metadata, build_generation_metadata


def main(config_path="assessment/configs/generate_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    rng = np.random.RandomState(seed)
    output_dir = Path(config["output_dir"])
    gt_dir = Path(config["ground_truth_dir"])
    default_sizes = [tuple(s) for s in config["default_sizes"]]
    max_size = config.get("max_size", None)  # optional cap on max dimension

    generators = get_all_generators()
    images_info = []
    total_count = 0

    for cat_name, gen_cls in sorted(generators.items()):
        cat_config = config.get("categories", {}).get(cat_name, {})
        if not cat_config.get("enabled", True):
            print(f"  Skipping disabled category: {cat_name}")
            continue

        sizes = cat_config.get("sizes_override")
        if sizes:
            sizes = [tuple(s) for s in sizes]
        else:
            sizes = default_sizes

        gen = gen_cls()
        variations = gen.get_variations()
        print(f"Category: {cat_name} ({len(variations)} variations, {len(sizes)} sizes)")

        for var in variations:
            # Use variation-specific sizes if provided, else category/default
            var_sizes = var.params.get("sizes_hint")
            if var_sizes:
                var_sizes = [tuple(s) for s in var_sizes]
            else:
                var_sizes = sizes

            # Apply max_size cap if configured
            if max_size:
                var_sizes = [s for s in var_sizes if max(s) <= max_size]
                if not var_sizes:
                    var_sizes = [(max_size, max_size)]
            for size in var_sizes:
                h, w = size
                fname = f"{var.name}_{w}x{h}.png"

                img_path = output_dir / cat_name / fname
                gt_path = gt_dir / cat_name / fname

                img_path.parent.mkdir(parents=True, exist_ok=True)
                gt_path.parent.mkdir(parents=True, exist_ok=True)

                image, mask = gen.generate(var, size, rng)

                cv2.imwrite(str(img_path), image)
                cv2.imwrite(str(gt_path), mask)

                images_info.append({
                    "filename": f"{cat_name}/{fname}",
                    "gt_filename": f"{cat_name}/{fname}",
                    "category": cat_name,
                    "variation": var.name,
                    "priority": var.priority,
                    "size": [h, w],
                    "params": var.params,
                })
                total_count += 1

    # Save metadata
    metadata = build_generation_metadata(config_path, seed, images_info)
    save_metadata(metadata, config["metadata_path"])

    print(f"\nGenerated {total_count} images.")
    print(f"Images:       {output_dir}")
    print(f"Ground truth: {gt_dir}")
    print(f"Metadata:     {config['metadata_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic assessment images")
    parser.add_argument("--config", default="assessment/configs/generate_config.yaml")
    args = parser.parse_args()
    main(args.config)
