"""Prepare test_dataset for pipeline: convert images to PNG, binarize+invert masks.

Usage:
    python -m assessment.prepare_test_dataset
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from .metadata.io import save_metadata


DATASET_DIR = Path("assessment/test_dataset")
OUTPUT_IMGS = Path("assessment/output/test_dataset_imgs")
OUTPUT_GT = Path("assessment/output/test_dataset_gt")
METADATA_PATH = OUTPUT_IMGS / "generation_metadata.json"

# Mapping: category -> (images_subdir, masks_subdir)
CATEGORIES = {
    "real": ("imgs", "masks"),
    "synthetic": ("images", "masks"),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def main():
    images_info = []

    for cat_name, (imgs_sub, masks_sub) in CATEGORIES.items():
        imgs_dir = DATASET_DIR / cat_name / imgs_sub
        masks_dir = DATASET_DIR / cat_name / masks_sub

        if not imgs_dir.exists():
            print(f"  Skipping {cat_name}: {imgs_dir} not found")
            continue

        out_imgs = OUTPUT_IMGS / cat_name
        out_gt = OUTPUT_GT / cat_name
        out_imgs.mkdir(parents=True, exist_ok=True)
        out_gt.mkdir(parents=True, exist_ok=True)

        img_files = [f for f in sorted(imgs_dir.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"Category '{cat_name}': {len(img_files)} images")

        for img_file in img_files:
            # Convert image to PNG
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  WARNING: cannot read {img_file}")
                continue

            out_name = img_file.stem + ".png"
            cv2.imwrite(str(out_imgs / out_name), img)

            # Find and process corresponding mask
            mask_file = None
            for ext in IMAGE_EXTENSIONS:
                candidate = masks_dir / (img_file.stem + ext)
                if candidate.exists():
                    mask_file = candidate
                    break

            if mask_file is not None:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Binarize (JPG artifacts)
                    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
                    # Invert: in test_dataset white=background, we need white=foreground
                    mask = 255 - mask
                    cv2.imwrite(str(out_gt / out_name), mask)
                    h, w = mask.shape
                    fg_pct = np.sum(mask > 127) / mask.size * 100
                    print(f"  {out_name}: {w}x{h}, GT fg={fg_pct:.1f}%")
                else:
                    print(f"  WARNING: cannot read mask {mask_file}")
            else:
                print(f"  WARNING: no mask found for {img_file.name}")

            images_info.append({
                "filename": f"{cat_name}/{out_name}",
                "gt_filename": f"{cat_name}/{out_name}",
                "category": cat_name,
                "variation": img_file.stem,
                "priority": "primary",
                "size": [img.shape[0], img.shape[1]],
                "params": {"source": "test_dataset", "original_format": img_file.suffix},
            })

    # Save metadata
    metadata = {
        "generated_at": "test_dataset",
        "config_path": "assessment/test_dataset",
        "seed": 0,
        "images": images_info,
    }
    save_metadata(metadata, str(METADATA_PATH))

    print(f"\nPrepared {len(images_info)} images.")
    print(f"Images: {OUTPUT_IMGS}")
    print(f"GT:     {OUTPUT_GT}")
    print(f"Meta:   {METADATA_PATH}")


if __name__ == "__main__":
    main()
