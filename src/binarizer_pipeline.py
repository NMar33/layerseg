import cv2
import numpy as np
import torch
import logging
from datetime import datetime
from pathlib import Path
from entities import BinarizerParams
from utils import setup_logging
from reports import save_report, save_full_report
from binarizers import load_model, make_seg, load_prep_img

logger = logging.getLogger("binarizer." + __name__)


def binarizer_pipeline(binarizer_params: BinarizerParams) -> None:

    logger.debug("started binarizer_pipeline")    
    start_time = datetime.now()  
    
    imgs_names = [img.name for img in Path(binarizer_params.path_imgs_dir).iterdir() if img.name != '.gitkeep']
    logger.debug("imgs_names: %s", imgs_names)

    if torch.cuda.is_available() and binarizer_params.device == "cuda":
        device = torch.device("cuda")
        logger.info("\n* Set up device to GPU (CUDA)")
    else:
        device = torch.device("cpu")
        logger.info("\n* Set up device to CPU")
    
    final_report = []
    # load model
    seg_model = load_model(binarizer_params.path_models_dir, binarizer_params.model_name, device)
    for img_name in imgs_names:
        logger.info("\n* Started processing image: " + img_name)
        # load original image
        # load array of imgs with different preprocessing methods (e.g. different size and scale)
        # load array of descriptions of preprocessing methods
        img_orig, size_orig, imgs_prep, \
            imgs_p_desc_shrt, imgs_p_desc_fll \
                = load_prep_img(binarizer_params, img_name)
        
        seg_masks = []
        for img_prep in imgs_prep:
            # make model specific preprocessing (e.g. add smart layers),
            # obtain segmentation mask and 
            # make model specific postprocessing
            img_seg = make_seg(binarizer_params, seg_model, img_prep, size_orig, device)
            seg_masks.append(img_seg)
        
        # img_glob_rep, img_glob_rep_desc, data_glob_rep \
        #     = 
        plots4final_report = save_report(
            binarizer_params, img_name, img_orig, 
            imgs_prep, imgs_p_desc_shrt, imgs_p_desc_fll, seg_masks)

        final_report.extend(plots4final_report)
    save_full_report(binarizer_params, final_report)

    end_time = datetime.now()
    logger.info("\n* Finished processing images. Total time: " + str(end_time - start_time))
        




