import logging
import logging.config
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
from entities import BinarizerParams
import cv2

from .plot import plot_imgs, plot_imgs_with_mask
from .pdf_csv_report import create_final_report

TRSH_ROUND = 3

logger = logging.getLogger("binarizer." + __name__)

def save_img(img_to_save, dir_path, img_name):
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_BGR2RGB)
    cv2.imwrite(Path(dir_path, img_name).as_posix(), img_to_save)

def save_mask(binarizer_params: BinarizerParams, mask_to_save, dir_path, mask_name):
    if binarizer_params.color_interest == "black":
        mask_to_save = (1 - mask_to_save)
    mask_to_save = (mask_to_save * 255).astype(np.uint8)
    cv2.imwrite(Path(dir_path, mask_name).as_posix(), mask_to_save)

def save_report(
        binarizer_params: BinarizerParams, img_name: str, img_orig: np.ndarray,
        imgs_diff_prep: List[np.ndarray], imgs_p_desc_shrt: List[str], imgs_p_desc_fll: List[str],
        segm_masks: List[np.ndarray]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """Saves report for image with different preprocessing  methods after obtaining a segmentation mask.

    Args:
        binarizer_params (BinarizerParams): parameters of the binarizer.
        img_name (str): name of the image.
        imgs_diff_prep (List[np.ndarray]): list of images with different preprocessing methods.
        imgs_diff_prep_desc (List[str]): list of descriptions of images with different preprocessing methods.
        segm_masks (List[np.ndarray]): list of segmentation masks.

    Returns:
        Tuple[np.ndarray, str, Dict[str, Any]]: global representation of the image, description of the global representation, data of the global representation.
    """
    logger.debug("started save_report")
    dpi = binarizer_params.report_dpi
    fig_sz = binarizer_params.report_fig_sz
    imgs_in_row = binarizer_params.imgs_in_row
    FIG = 1 # figure start number

    # setup report dir
    report_dir = binarizer_params.path_report_dir
    time_now = datetime.now().strftime("%d_%H_%M")
    report_name = f"{binarizer_params.report_name}_{img_name}_{time_now}"
    path_full_rep_dir = Path(report_dir, report_name).as_posix()
    path_mask_dir = Path(path_full_rep_dir, "masks").as_posix()


    # create report dirs
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(path_full_rep_dir).mkdir(parents=True, exist_ok=True)
    Path(path_mask_dir).mkdir(parents=True, exist_ok=True)

    if binarizer_params.report_short:
        short_report_dir = binarizer_params.short_report_dir
        full_path_short_rep_dir = Path(short_report_dir, report_name).as_posix()
        Path(full_path_short_rep_dir).mkdir(parents=True, exist_ok=True)

    
    fig = FIG
    plots4final_report = []

    imgs = [img_orig, *imgs_diff_prep]
    labels = ["Original image", *imgs_p_desc_shrt]
    plot_org_and_p_imgs_name = f"{fig:02d}_preprocessed_imgs.png"
    plot_org_and_p_imgs = plot_imgs(imgs, labels, dpi, fig_sz, suptitle=f"{img_name} Original and preprocessed images", img_cmaps=["rgb", "gray"])
    save_img(plot_org_and_p_imgs, path_full_rep_dir, plot_org_and_p_imgs_name)

    plots4final_report.append(plot_org_and_p_imgs)

    for i, img_prep in enumerate(imgs_diff_prep):
        fig += 1
        mask = segm_masks[i]
        desc_short = imgs_p_desc_shrt[i]
        desc = imgs_p_desc_fll[i]
        segm_plot_names = []
        masks_plot = []
        labels = []
        masks_name = f"{fig:02d}{0}_{desc_short}_soft_bin"
        save_mask(binarizer_params, mask, path_mask_dir, f"{img_name};{masks_name}.png")
        
        if binarizer_params.report_short:
            save_mask(binarizer_params, mask, full_path_short_rep_dir, f"{img_name};{masks_name}.png")

        for j, thrsh in enumerate(binarizer_params.binarizer_thresholds):
            thrsh = np.round(thrsh, TRSH_ROUND)
            segm_masks_name = f"{fig:02d}{j + 1}_{desc_short}_threshold_{thrsh}"
            labels.append(f"{desc_short} threshold={thrsh}")
            segm_plot_names.append(segm_masks_name)
            mask_thrsh = mask.copy()
            mask_thrsh[mask_thrsh < thrsh] = 0
            mask_thrsh[mask_thrsh >= thrsh] = 1
            masks_plot.append(mask_thrsh)
            save_mask(binarizer_params, mask_thrsh, path_mask_dir, f"{img_name};{segm_masks_name}.png")
        


        for batch_start in range(0, len(masks_plot), imgs_in_row):
            plot_org_w_masks = plot_imgs_with_mask(
                img_orig, masks_plot[batch_start:(batch_start + imgs_in_row)],
                ["Original image", *labels[batch_start:(batch_start + imgs_in_row)]],
                dpi, fig_sz, suptitle=desc.replace("\n", " "), show_img=False)
            save_img(plot_org_w_masks, path_full_rep_dir, segm_plot_names[batch_start] + ".png")
            plots4final_report.append(plot_org_w_masks)
    return plots4final_report


def save_full_report(binarizer_params: BinarizerParams, plots4final_report):
    logger.debug("Start save_full_report")
    
    report_dir = binarizer_params.path_report_dir
    time_now = datetime.now().strftime("%d_%H_%M")
    full_report_name = f"{binarizer_params.report_name}_{time_now}"
    
    create_final_report(report_dir, full_report_name, plots4final_report) 
    
    if binarizer_params.report_short:
        short_report_dir = binarizer_params.short_report_dir
        create_final_report(short_report_dir, full_report_name, plots4final_report) 