"""Console and file logging for experiment runs."""

import logging
from pathlib import Path


def get_log(log_path, save_log_name):
    directory = Path(log_path)
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / (save_log_name + ".log")
    logger = logging.getLogger(str(logfile.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    for handler in [logging.StreamHandler(), logging.FileHandler(logfile, mode="w")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
