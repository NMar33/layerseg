import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import click
import logging
from entities import read_binarizer_params
from utils import setup_logging
from binarizer_pipeline import binarizer_pipeline


@click.command(
    name="binarizer_pipeline",
    help="""""",
)
@click.option(
    "-cfg",
    "--config_path",
    default="configs/config.yaml",
    type=str,
    prompt="Please, enter the config file path.",
    help="Specify the path to the config file.",
)
def binarizer_pipeline_command(config_path: str) -> None:
    binarizer_params = read_binarizer_params(config_path)
    setup_logging(binarizer_params.path_logging_config)
    logger = logging.getLogger("binarizer." + __name__)
    logger.info("\n" + "#" * 50)
    logger.info("started program with params %s", binarizer_params)
    logger.info("\n" + "#" * 50)
    binarizer_pipeline(binarizer_params)

if __name__ == "__main__":
    binarizer_pipeline_command()