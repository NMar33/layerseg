import torch
import logging
from pathlib import Path
import cv2
from entities import BinarizerParams
from .legacy import UNet

logger = logging.getLogger("binarizer." + __name__)

def load_model(path_models_dir, model_name, device):
    model_path = Path(path_models_dir, model_name).as_posix()
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def scale_factor_prep(img2prep, scale_factors, descr_short, descr_full):
    h, w = img2prep.shape
    descr_short_out, descr_full_out = [], []
    img_prep_out = []
    for sf in scale_factors:
        size_cv = (int(w * sf), int(h * sf))
        img_prep = cv2.resize(img2prep, size_cv)
        img_prep_out.append(img_prep)
        descr_short_out.append(descr_short + f"scale_{sf}")
        descr_full_out.append(descr_full + f" (scale factor: {sf})")
    return img_prep_out, descr_short_out, descr_full_out

def arrays_extend(arrays, arrays_to_add):
    for i in range(len(arrays)):
        arrays[i].extend(arrays_to_add[i])
    return arrays

def load_prep_img(binarizer_params: BinarizerParams, img_name):
    logger.debug(f"Loading and preprocessing image: {img_name}")
    img_orig = cv2.imread(Path(binarizer_params.path_imgs_dir, img_name).as_posix())
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    size_orig = img_orig.shape[:2]

    imgs_prep_out, imgs_desc_short_out, imgs_desc_full_out = [], [], []

    # load image based on configured input mode
    if binarizer_params.input_mode == "grayscale":
        img2prep = cv2.imread(Path(binarizer_params.path_imgs_dir, img_name).as_posix(), cv2.IMREAD_GRAYSCALE)
    elif binarizer_params.input_mode == "rgb":
        img2prep = cv2.imread(Path(binarizer_params.path_imgs_dir, img_name).as_posix())
        img2prep = cv2.cvtColor(img2prep, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unknown input_mode: {binarizer_params.input_mode}")

    scale_factors = binarizer_params.scale_factors
    i_p_out, d_s_out, d_f_out = scale_factor_prep(
        img2prep, scale_factors,
        "", f"{img_name} \n")
    arrays_extend((imgs_prep_out, imgs_desc_short_out, imgs_desc_full_out), (i_p_out, d_s_out, d_f_out))

    if binarizer_params.gaussian_blur == True:
        ks = binarizer_params.gaussian_blur_kernel_size
        img_prep = cv2.GaussianBlur(img2prep, (ks, ks), 0)
        i_p_out, d_s_out, d_f_out = scale_factor_prep(img_prep, scale_factors, f"gblur_{ks}_", f"{img_name} \n(gaussian blur)")
        arrays_extend((imgs_prep_out, imgs_desc_short_out, imgs_desc_full_out), (i_p_out, d_s_out, d_f_out))

    return img_orig, size_orig, imgs_prep_out, imgs_desc_short_out, imgs_desc_full_out
