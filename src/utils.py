import torch
from torch import nn
import numpy as np
import os
import logging
from contextlib import contextmanager


def set_random_seed(seed: int = 1996):
    """
    Passes random seed to the code for easy code reproductiblity.
    seed: int = a random seed number
    """
    torch.random.seed(seed)
    np.random.seed(seed)
    os.environ["PTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministis = True
    torch.backends.cudnn.benchmark = True


def get_logger(outfile: str = None):
    logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    if outfile is not None:
        fh = logging.FileHandler(outfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    logger.info("logger set up")
    return logger


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)