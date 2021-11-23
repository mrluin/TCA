import torch
import numpy as np

from utils.matlab_functions import bgr2ycbcr


class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """
    def __init__(self):
        self.initialized = False
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return self.avg

    @property
    def get_sum(self):
        return self.sum


def reorder_image(img, input_order='HWC'):
    """ Reorder images to 'HWC' order

    (h, w) -> (h, w, 1)
    (c, h, w) -> (h, w, c)
    (h, w, c) -> (h, w, c)

    Args:
        img: input image
        input_order
    Returns:
        reodered image
    """
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'"
        )
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)

    return img


def to_y_channel(img):
    """ Change to Y channel of YCbCr

    Args:
        img: input image with range [0, 255]

    Returns:
        Images with range [0, 255] float type without round
    """

    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255
