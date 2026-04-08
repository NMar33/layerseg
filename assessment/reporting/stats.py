"""Compute statistics for binary masks and soft masks."""
import numpy as np
import cv2


def mask_area_stats(mask, foreground_is_black=True):
    """Compute area statistics for a binary mask (0/255 values).

    Args:
        mask: uint8 array with 0 and 255 values
        foreground_is_black: if True, dark pixels (0) are foreground (pipeline convention)

    Returns dict with percent_foreground, percent_background, num_components, etc.
    """
    if foreground_is_black:
        binary = (mask <= 127).astype(np.uint8)  # dark = foreground
    else:
        binary = (mask > 127).astype(np.uint8)   # white = foreground
    total = mask.size
    fg = int(np.sum(binary))

    stats = {
        "percent_foreground": round(fg / total * 100, 2),
        "percent_background": round((total - fg) / total * 100, 2),
    }

    # Connected components
    num_labels, labels = cv2.connectedComponents(binary)
    num_fg_components = num_labels - 1  # subtract background
    stats["num_components"] = num_fg_components

    if num_fg_components > 0:
        component_sizes = []
        for label_id in range(1, num_labels):
            component_sizes.append(int(np.sum(labels == label_id)))
        stats["largest_component_frac"] = round(max(component_sizes) / total * 100, 2)
        stats["mean_component_size"] = round(np.mean(component_sizes), 1)
    else:
        stats["largest_component_frac"] = 0.0
        stats["mean_component_size"] = 0.0

    return stats


def soft_mask_stats(soft_mask):
    """Compute statistics for a soft (probability) mask (float 0-1 or uint8 0-255).

    Returns dict with mean_prob, std_prob.
    """
    if soft_mask.dtype == np.uint8:
        probs = soft_mask.astype(np.float64) / 255.0
    else:
        probs = soft_mask.astype(np.float64)

    return {
        "mean_prob": round(float(np.mean(probs)), 4),
        "std_prob": round(float(np.std(probs)), 4),
    }


def compare_with_gt(predicted_mask, gt_mask):
    """Compute IoU, Dice, Precision, Recall between predicted and ground truth.

    Both masks should be uint8 0/255.
    Pipeline saves with color_interest='black': lines=BLACK(0), bg=WHITE(255).
    GT masks: lines=WHITE(255), bg=BLACK(0).
    So we invert the predicted mask to match GT convention.

    Returns dict with iou, dice, precision, recall.
    """
    pred = (predicted_mask <= 127).astype(bool)  # dark pixels = detected foreground
    gt = (gt_mask > 127).astype(bool)            # white pixels = GT foreground

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    pred_sum = np.sum(pred)
    gt_sum = np.sum(gt)

    iou = intersection / union if union > 0 else 1.0
    dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 1.0
    precision = intersection / pred_sum if pred_sum > 0 else 1.0
    recall = intersection / gt_sum if gt_sum > 0 else 1.0

    return {
        "iou": round(float(iou), 4),
        "dice": round(float(dice), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
    }
