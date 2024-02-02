import logging
import re

LOG_FILE = 'run.log'

def obtain_module_name(full_filepath: str) -> str:
    regex = r"([^.]+\.)+"
    return re.sub(regex, "", full_filepath)

def configure_logging(name, log_file=LOG_FILE, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter("%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(ch)

    return logger