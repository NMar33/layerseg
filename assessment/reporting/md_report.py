"""Generate Markdown reports from pipeline results."""
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .stats import mask_area_stats, compare_with_gt
from ..metadata.io import load_metadata


def generate_reports(config):
    """Generate brief and full Markdown reports.

    Args:
        config: dict from report_config.yaml
    """
    pipeline_results_dir = Path(config["pipeline_results_dir"])
    gt_dir = Path(config["ground_truth_dir"])
    output_dir = Path(config["output_dir"])
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    gen_meta = load_metadata(config["generation_metadata"])
    pipe_meta = load_metadata(config["pipeline_metadata"])

    dpi = config.get("report_dpi", 150)
    fig_size = config.get("fig_size", 4)

    # Index generation metadata by filename
    img_meta = {}
    for entry in gen_meta["images"]:
        img_meta[entry["filename"]] = entry

    # Group pipeline results by preset
    results_by_preset = {}
    for r in pipe_meta["results"]:
        preset = r["preset"]
        results_by_preset.setdefault(preset, []).append(r)

    # For each preset, find mask files in output directories
    report_data = _collect_report_data(
        pipeline_results_dir, gt_dir, img_meta, results_by_preset
    )

    # Generate comparison images
    _generate_comparison_images(report_data, imgs_dir, dpi, fig_size)

    # Write reports
    _write_report(output_dir / "full_report.md", report_data, imgs_dir, mode="full")
    _write_report(output_dir / "brief_report.md", report_data, imgs_dir, mode="brief")

    n_full = sum(len(items) for items in report_data.values())
    n_brief = sum(
        len([it for it in items if it["priority"] == "primary"])
        for items in report_data.values()
    )
    print(f"Full report:  {output_dir / 'full_report.md'} ({n_full} entries)")
    print(f"Brief report: {output_dir / 'brief_report.md'} ({n_brief} entries)")


def _collect_report_data(pipeline_results_dir, gt_dir, img_meta, results_by_preset):
    """Collect all data needed for reports.

    Returns dict keyed by category, each value is list of entry dicts.
    """
    report_data = {}

    # Use "no_blur" preset if available, else first preset
    if "no_blur" in results_by_preset:
        primary_preset = "no_blur"
    else:
        primary_preset = list(results_by_preset.keys())[0] if results_by_preset else None
    if not primary_preset:
        return report_data

    for result in results_by_preset[primary_preset]:
        output_dir = Path(result["output_dir"])
        input_dir = Path(result["input_dir"])
        cat_name = input_dir.name

        # Find all input images
        for img_file in sorted(input_dir.glob("*.png")):
            img_key = f"{cat_name}/{img_file.name}"
            meta = img_meta.get(img_key, {})
            priority = meta.get("priority", "secondary")

            # Find GT mask
            gt_path = gt_dir / cat_name / img_file.name
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else None

            # Find pipeline output masks
            masks_info = _find_masks(output_dir, img_file.name)

            entry = {
                "img_name": img_file.name,
                "img_path": str(img_file),
                "category": cat_name,
                "priority": priority,
                "variation": meta.get("variation", img_file.stem),
                "size": meta.get("size", [0, 0]),
                "params": meta.get("params", {}),
                "gt_path": str(gt_path) if gt_mask is not None else None,
                "masks": masks_info,
                "gt_mask": gt_mask,
                "preset": primary_preset,
                "settings": result["settings"],
            }

            # Compute stats for each mask
            if masks_info and gt_mask is not None:
                for m in masks_info:
                    pred = cv2.imread(m["path"], cv2.IMREAD_GRAYSCALE)
                    if pred is not None:
                        m["area_stats"] = mask_area_stats(pred)
                        m["gt_comparison"] = compare_with_gt(pred, gt_mask)

            report_data.setdefault(cat_name, []).append(entry)

    return report_data


def _find_masks(output_dir, img_name):
    """Find generated mask files for a given input image in pipeline output.

    Searches recursively since output may be nested (sf_X/report_.../masks/).
    """
    masks = []
    output_path = Path(output_dir)

    # Find all masks/ dirs recursively
    for masks_dir in sorted(output_path.rglob("masks")):
        if not masks_dir.is_dir():
            continue

        # Check if any mask file contains the image name
        found = False
        for mask_file in sorted(masks_dir.glob("*.png")):
            if img_name not in mask_file.name:
                continue
            found = True
            label = mask_file.stem
            threshold = None
            if "threshold_" in label:
                try:
                    threshold = float(label.split("threshold_")[1])
                except (ValueError, IndexError):
                    pass

            is_soft = "soft_bin" in label

            masks.append({
                "path": str(mask_file),
                "filename": mask_file.name,
                "label": label,
                "threshold": threshold,
                "is_soft": is_soft,
            })

        if found:
            break  # Use first matching masks dir

    return masks


def _generate_comparison_images(report_data, imgs_dir, dpi, fig_size):
    """Generate side-by-side comparison PNGs for the report."""
    for cat_name, entries in report_data.items():
        cat_dir = imgs_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        for entry in entries:
            _make_comparison_image(entry, cat_dir, dpi, fig_size)


def _make_comparison_image(entry, out_dir, dpi, fig_size):
    """Create comparison: Input | GT | Soft mask | t=0.5 | t=0.7."""
    img = cv2.imread(entry["img_path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    gt = entry.get("gt_mask")
    masks = entry.get("masks", [])

    # Find soft mask
    soft_mask_info = None
    for m in masks:
        if m.get("is_soft"):
            soft_mask_info = m
            break

    # Find specific threshold masks (t=0.5 and t=0.7)
    target_thresholds = [0.5, 0.7]
    threshold_masks = []
    for target_t in target_thresholds:
        best = None
        best_dist = float("inf")
        for m in masks:
            if m.get("threshold") is not None:
                dist = abs(m["threshold"] - target_t)
                if dist < best_dist:
                    best_dist = dist
                    best = m
        if best and best_dist < 0.05:  # within 0.05 of target
            threshold_masks.append(best)

    # Count panels: Input + GT + Soft + threshold masks
    n_panels = 1 + (1 if gt is not None else 0) + (1 if soft_mask_info else 0) + len(threshold_masks)
    if n_panels < 2:
        n_panels = 2  # minimum

    h, w = img.shape[:2]
    ratio = h / w
    fig, axes = plt.subplots(1, n_panels,
        figsize=(fig_size * n_panels, fig_size * ratio * 1.15), dpi=dpi)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    axes[idx].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[idx].set_title("Input")
    axes[idx].axis("off")
    idx += 1

    if gt is not None:
        axes[idx].imshow(gt, cmap="gray", vmin=0, vmax=255)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")
        idx += 1

    if soft_mask_info:
        soft_img = cv2.imread(soft_mask_info["path"], cv2.IMREAD_GRAYSCALE)
        if soft_img is not None:
            axes[idx].imshow(soft_img, cmap="gray", vmin=0, vmax=255)
            axes[idx].set_title("Soft output")
            axes[idx].axis("off")
            idx += 1

    for m in threshold_masks:
        mask_img = cv2.imread(m["path"], cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            axes[idx].imshow(mask_img, cmap="gray", vmin=0, vmax=255)
            t = m.get("threshold", "?")
            iou = m.get("gt_comparison", {}).get("iou", "?")
            axes[idx].set_title(f"t={t}\nIoU={iou}")
            axes[idx].axis("off")
            idx += 1

    plt.suptitle(f"{entry['variation']} ({entry['size'][1]}x{entry['size'][0]})", fontsize=10)
    plt.tight_layout()

    out_path = out_dir / f"{entry['img_name'].replace('.png', '_comparison.png')}"
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    entry["comparison_img"] = str(out_path)


def _write_report(report_path, report_data, imgs_dir, mode="full"):
    """Write a Markdown report.

    mode: "full" (all entries) or "brief" (primary only)
    """
    lines = []
    lines.append(f"# Bin4_PhasIAn Assessment Report ({mode.title()})")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Table of contents
    lines.append("## Contents\n")
    for cat_name in sorted(report_data.keys()):
        entries = report_data[cat_name]
        if mode == "brief":
            entries = [e for e in entries if e["priority"] == "primary"]
        if entries:
            lines.append(f"- [{cat_name}](#{cat_name}) ({len(entries)} images)")
    lines.append("")

    # Per-category sections
    for cat_name in sorted(report_data.keys()):
        entries = report_data[cat_name]
        if mode == "brief":
            entries = [e for e in entries if e["priority"] == "primary"]
        if not entries:
            continue

        lines.append(f"## {cat_name}\n")

        # Summary table
        lines.append("| Variation | Size | Threshold | IoU | Dice | FG % |")
        lines.append("|-----------|------|-----------|-----|------|---------|")

        for entry in entries:
            # Find best IoU mask
            best_mask = None
            best_iou = -1
            for m in entry.get("masks", []):
                gt_comp = m.get("gt_comparison", {})
                if gt_comp.get("iou", -1) > best_iou:
                    best_iou = gt_comp["iou"]
                    best_mask = m

            if best_mask:
                gt_c = best_mask.get("gt_comparison", {})
                area = best_mask.get("area_stats", {})
                lines.append(
                    f"| {entry['variation']} | {entry['size'][1]}x{entry['size'][0]} "
                    f"| {best_mask.get('threshold', '-')} "
                    f"| {gt_c.get('iou', '-')} | {gt_c.get('dice', '-')} "
                    f"| {area.get('percent_foreground', '-')} |"
                )
            else:
                lines.append(
                    f"| {entry['variation']} | {entry['size'][1]}x{entry['size'][0]} "
                    f"| - | - | - | - |"
                )

        lines.append("")

        # Per-entry images
        for entry in entries:
            lines.append(f"### {entry['variation']} ({entry['size'][1]}x{entry['size'][0]})\n")

            comp_img = entry.get("comparison_img")
            if comp_img:
                rel_path = os.path.relpath(comp_img, report_path.parent).replace("\\", "/")
                lines.append(f"![{entry['variation']}]({rel_path})\n")

            # Threshold sweep table (full report only)
            if mode == "full" and entry.get("masks"):
                threshold_masks = [m for m in entry["masks"] if m.get("threshold") is not None]
                if threshold_masks:
                    lines.append("| Threshold | IoU | Dice | Precision | Recall | FG % |")
                    lines.append("|-----------|-----|------|-----------|--------|---------|")
                    for m in threshold_masks:
                        gt_c = m.get("gt_comparison", {})
                        area = m.get("area_stats", {})
                        lines.append(
                            f"| {m.get('threshold', '-')} "
                            f"| {gt_c.get('iou', '-')} | {gt_c.get('dice', '-')} "
                            f"| {gt_c.get('precision', '-')} | {gt_c.get('recall', '-')} "
                            f"| {area.get('percent_foreground', '-')} |"
                        )
                    lines.append("")

        lines.append("---\n")

    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
