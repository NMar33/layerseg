import torch
import cv2
from torchvision import transforms as T
from entities import BinarizerParams
from .legacy import make_img_with_smart_layers, make_img_with_smart_layers_cached

def seg_post_m220805(img_seg):
    img_seg = 1 / (1 + torch.exp(img_seg[0] - img_seg[1]))
    return img_seg.squeeze().detach().cpu().numpy()

def make_seg(binarizer_params: BinarizerParams, seg_model, img_prep, size_orig, device):
    transform = T.ToTensor()
    with torch.no_grad():
        img_prep = transform(img_prep)

        # preprocessing (config-driven)
        if binarizer_params.preprocessing == "smart_contrast":
            if binarizer_params.cache == True:
                make_smart_layers_fn = make_img_with_smart_layers_cached(binarizer_params.cache_dir)
            else:
                make_smart_layers_fn = make_img_with_smart_layers
            img_prep = make_smart_layers_fn(img_prep)
            img_seg = seg_model(img_prep.to(device).unsqueeze(dim=0))
        else:
            raise ValueError(f"Unknown preprocessing: {binarizer_params.preprocessing}")

        # postprocessing (config-driven)
        if binarizer_params.postprocessing == "sigmoid_diff":
            img_seg = seg_post_m220805(img_seg.squeeze())
        else:
            raise ValueError(f"Unknown postprocessing: {binarizer_params.postprocessing}")

        size_orig_cv = (size_orig[1], size_orig[0])
        img_seg = cv2.resize(img_seg, size_orig_cv)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return img_seg
