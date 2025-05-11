import functools
import time
import logging
import inspect
from typing import Callable, Optional
import os
import re
from datetime import datetime


def is_valid_filepath(file_path: str) -> bool:
    """
    Check if the given string is in a valid file path format.
    Ensures the path ends with a filename and extension.
    """
    # Remove any surrounding quotes
    file_path = file_path.strip("\"'")

    # Check for invalid characters in the path
    invalid_chars = r'[<>:"|?*]'  # Invalid characters in Windows
    if re.search(invalid_chars, file_path):
        return False

    # Check for valid path format with filename and extension
    # Pattern explanation:
    # ^ - start of string
    # [a-zA-Z0-9\s\.\-_\\\/]+ - one or more valid path characters
    # [^\\\/] - not a slash (ensures not ending with slash)
    # \. - literal dot
    # [a-zA-Z0-9]+ - one or more alphanumeric characters for extension
    # $ - end of string
    valid_path_pattern = r"^[a-zA-Z0-9\s\.\-_\\\/]+[^\\\/]\.[a-zA-Z0-9]+$"
    return bool(re.match(valid_path_pattern, file_path))


def get_logger(name, level=logging.INFO, file_name=None):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.propagate = False
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if file_name:
        if is_valid_filepath(file_name):
            file_handler = logging.FileHandler(file_name)
        else:
            os.makedirs(file_name, exist_ok=True)
            name = f"{os.path.basename(file_name)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            file_handler = logging.FileHandler(os.path.join(file_name, name))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_execution_time(logger=get_logger(__name__), level=logging.INFO):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = fn(*args, **kwargs)
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            # Get function metadata
            module_name = fn.__module__

            logger.log(
                level,
                f"Function '{fn.__name__}' from module '{module_name}' executed in {execution_time:.4f} seconds",
            )
            return result

        return wrapper

    return decorator
