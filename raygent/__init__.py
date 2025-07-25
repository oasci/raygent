# This file is licensed under the Prosperity Public License 3.0.0.
# You may use, copy, and share it for noncommercial purposes.
# Commercial use is allowed for a 30-day trial only.
#
# Contributor: Scienting Studio
# Source Code: https://github.com/scienting/simlify
#
# See the LICENSE.md file for full license terms.


"""Parallelism, Delegated"""

import os
import sys
from ast import literal_eval

from loguru import logger

from .batch import batch_generator, BatchMessage
from .task import Task

__all__ = ["batch_generator", "BatchMessage", "Task"]

logger.disable("raygent")

LOG_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)


def enable_logging(
    level_set: int,
    stdout_set: bool = True,
    file_path: str | None = None,
    log_format: str = LOG_FORMAT,
) -> None:
    r"""Enable logging.

    Args:
        level: Requested log level: `10` is debug, `20` is info.
        file_path: Also write logs to files here.
    """
    config: dict[str, list[dict[str, object]]] = {"handlers": []}
    if stdout_set:
        config["handlers"].append(
            {
                "sink": sys.stdout,
                "level": level_set,
                "format": log_format,
                "colorize": True,
            }
        )
    if isinstance(file_path, str):
        config["handlers"].append(
            {
                "sink": file_path,
                "level": level_set,
                "format": log_format,
                "colorize": False,
            }
        )
    # https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger.configure
    logger.configure(**config)

    logger.enable("raygent")


if literal_eval(os.environ.get("RAYGENT_LOG", "False")):
    level = int(os.environ.get("RAYGENT_LOG_LEVEL", 20))
    stdout = literal_eval(os.environ.get("RAYGENT_STDOUT", "True"))
    log_file_path = os.environ.get("RAYGENT_LOG_FILE_PATH", None)
    enable_logging(level, stdout, log_file_path)
