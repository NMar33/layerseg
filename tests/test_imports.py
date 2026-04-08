"""Test that all project imports work correctly."""


def test_import_torch():
    import torch
    assert hasattr(torch, "__version__")


def test_import_entities():
    from entities import BinarizerParams, read_binarizer_params
    assert BinarizerParams is not None
    assert callable(read_binarizer_params)


def test_import_binarizers():
    from binarizers import load_model, load_prep_img, make_seg
    from binarizers import UNet, DoubleConv, Down, Up, OutConv
    assert callable(load_model)
    assert callable(load_prep_img)
    assert callable(make_seg)


def test_import_reports():
    from reports import save_report, save_full_report
    assert callable(save_report)
    assert callable(save_full_report)


def test_import_utils():
    from utils import setup_logging
    assert callable(setup_logging)


def test_import_pipeline():
    from binarizer_pipeline import binarizer_pipeline
    assert callable(binarizer_pipeline)
