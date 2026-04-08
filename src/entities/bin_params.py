from dataclasses import dataclass
from typing import List, Tuple
from marshmallow_dataclass import class_schema

import yaml

@dataclass()
class BinarizerParams:
    # path to images for binarization
    path_imgs_dir: str
    path_report_dir: str
    path_logging_config: str
    # pretrained model path
    path_models_dir: str
    model_name: str
    # if True, then cache will be used
    cache: bool
    cache_dir: str

    # report name
    report_name: str
    # scale factors for preprocessing
    scale_factors: List[float]
    # if True, then gaussian blur will be performed
    gaussian_blur: bool
    gaussian_blur_kernel_size: int
    # thresholds for binarization (report with different thresholds will be created)
    binarizer_thresholds: List[float]
    # color of the original image
    original_img_color_map: str # "gray" or "rgb"
    # amount of images in row in report
    imgs_in_row: int
    color_interest: str # "white" or "black"

    # dpi of images in report
    report_dpi: int
    # size of images in report
    report_fig_sz: int
    report_short: bool
    short_report_dir: str

    # computing device: "cpu" or "cuda"
    device: str

    # image loading mode: "grayscale" or "rgb"
    input_mode: str = "grayscale"
    # preprocessing applied before model: "smart_contrast" creates 3-channel tensor from grayscale
    preprocessing: str = "smart_contrast"
    # postprocessing applied to model output: "sigmoid_diff" computes sigmoid(ch0 - ch1)
    postprocessing: str = "sigmoid_diff"
    # number of input channels the model expects
    n_channels: int = 3
    # number of output classes from the model
    n_classes: int = 2
    


BinarizerParamsSchema = class_schema(BinarizerParams)


def read_binarizer_params(path: str) -> BinarizerParams:
    with open(path, "r") as input_stream:
        schema = BinarizerParamsSchema()
        return schema.load(yaml.safe_load(input_stream))