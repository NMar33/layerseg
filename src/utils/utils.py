import logging
import logging.config
import yaml


def setup_logging(logging_config_path: str) -> None:
    if logging_config_path != "default":        
        with open(logging_config_path, "r") as f:
            log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
    else:
        logger = logging.getLogger("binarizer")
        handler = logging.StreamHandler()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)