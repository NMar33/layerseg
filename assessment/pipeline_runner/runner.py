"""Wrapper to run binarizer pipeline programmatically on assessment images."""
import time
import logging
from pathlib import Path

import cv2

from .path_setup import setup_binarizer_imports


def get_scale_factors_for_size(img_path, size_based_scales):
    """Determine scale_factors based on image dimensions.

    Args:
        img_path: path to image file
        size_based_scales: dict mapping max_dim_threshold -> scale_factors list

    Returns:
        list of scale factors
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [1]
    max_dim = max(img.shape)

    # Sort thresholds and find first match
    for threshold in sorted(size_based_scales.keys()):
        if max_dim <= threshold:
            return size_based_scales[threshold]
    return [1]


def run_on_category(project_root, input_dir, output_dir, preset_name, preset_config, pipeline_config):
    """Run binarizer pipeline on all images in input_dir with given preset.

    Groups images by their scale_factors (based on size) and runs pipeline per group.

    Returns:
        list of result dicts (one per image group)
    """
    setup_binarizer_imports(project_root)

    from entities import BinarizerParams
    from binarizer_pipeline import binarizer_pipeline
    from utils import setup_logging

    logging_config = pipeline_config.get("logging_config", "default")
    setup_logging(logging_config)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    size_based_scales = {}
    for k, v in pipeline_config.get("size_based_scales", {256: [1]}).items():
        size_based_scales[int(k)] = v

    thresholds = pipeline_config.get("binarizer_thresholds", [0.5, 0.7])

    # Group images by scale_factors
    groups = {}  # scale_factors_tuple -> list of image paths
    input_path = Path(input_dir)
    img_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    img_files_all = []
    for ext in img_extensions:
        img_files_all.extend(input_path.glob(ext))
    for img_file in sorted(img_files_all):
        sf = tuple(get_scale_factors_for_size(img_file, size_based_scales))
        groups.setdefault(sf, []).append(img_file)

    results = []
    for scale_factors, img_files in groups.items():
        # Create temp dir with just these images (pipeline reads all from dir)
        sf_label = "_".join(f"{s}" for s in scale_factors)
        group_input = Path(output_dir) / f"_tmp_input_sf{sf_label}"
        group_input.mkdir(parents=True, exist_ok=True)

        # Symlink/copy images to temp dir
        for img_file in img_files:
            dst = group_input / img_file.name
            if not dst.exists():
                import shutil
                shutil.copy2(str(img_file), str(dst))

        group_output = Path(output_dir) / f"sf_{sf_label}"

        params = BinarizerParams(
            path_imgs_dir=str(group_input),
            path_report_dir=str(group_output),
            path_logging_config=logging_config,
            path_models_dir=str(Path(project_root) / pipeline_config["path_models_dir"]),
            model_name=pipeline_config["model_name"],
            cache=pipeline_config.get("cache", True),
            cache_dir=str(Path(project_root) / pipeline_config.get("cache_dir", "cache")),
            report_name=f"assess_{preset_name}",
            scale_factors=list(scale_factors),
            gaussian_blur=preset_config.get("gaussian_blur", False),
            gaussian_blur_kernel_size=preset_config.get("gaussian_blur_kernel_size", 5),
            binarizer_thresholds=thresholds,
            original_img_color_map="gray",
            imgs_in_row=3,
            color_interest="black",
            report_dpi=150,
            report_fig_sz=4,
            report_short=False,
            short_report_dir=str(Path(output_dir) / "short"),
            device=pipeline_config.get("device", "cpu"),
            input_mode=pipeline_config.get("input_mode", "grayscale"),
            preprocessing=pipeline_config.get("preprocessing", "smart_contrast"),
            postprocessing=pipeline_config.get("postprocessing", "sigmoid_diff"),
            n_channels=pipeline_config.get("n_channels", 3),
            n_classes=pipeline_config.get("n_classes", 2),
        )

        start = time.time()
        binarizer_pipeline(params)
        elapsed = time.time() - start

        results.append({
            "preset": preset_name,
            "input_dir": str(input_dir),
            "output_dir": str(group_output),
            "scale_factors": list(scale_factors),
            "n_images": len(img_files),
            "settings": {
                **preset_config,
                "scale_factors": list(scale_factors),
                "binarizer_thresholds": thresholds,
            },
            "processing_time_seconds": round(elapsed, 2),
        })

        # Clean up temp input dir
        import shutil
        shutil.rmtree(str(group_input), ignore_errors=True)

    return results
