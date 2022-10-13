import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.traceback import install
from rich.tree import Tree


def get_logger(name=__name__, set_default_rich_handler=False) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    if set_default_rich_handler:
        logger.setLevel(logging.INFO)
        ch = RichHandler()
        ch.setLevel(logging.INFO)

        # create formatter
        formatter = logging.Formatter("%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    install()

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup

    return logger


def demo_logger():
    logger = get_logger(__name__, set_default_rich_handler=True)

    logger.info("Hello World")
    logger.debug("Debugging")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
    logger.exception("Exception")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretty_config(
    config: DictConfig,
    resolve: bool = True,
):
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    for field in config.keys():
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    return tree
