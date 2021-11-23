import cv2
import math

import numpy
import numpy as np
import os
import torch
from torchvision.utils import make_grid

__all__ = [
    'img2tensor',
    'tensor2img',
    'imfrombytes',
    'imwrite',
    'crop_border',
    'tensor_sift_warp',
    'tensor_sift_warp_from_homo'
]


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """ Numpy array to tensor
    Args:
        imgs
        bgr2rgb: whether to change bgr to rgb
        float32: whether to change to float32
    Returns:
        list[tensor] | tensor
    """
    def _totensor(img, bgr2rgb, float32):
        # if len(img.shape) == 2:
        #     img = torch.from_numpy(img)
        #     if float32:
        #         img = img.float()
        #     return img
        if len(img.shape) == 3:
            if img.shape[2] == 3 and bgr2rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img.transpose(2, 0, 1))
        else:
            assert len(img.shape) == 2
            img = torch.from_numpy(img).unsqueeze(0)

        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """ Convert torch tensor into image numpy array

    After clamping to [min, max], values will be normalized to [0, 1]
    Args:
        tensor accept shapes:
        1) 4D mini-batch tensor of shape (B x 3/1 x H x W)
        2) 3D tensor of shape (3/1 x H x W)
        3) 2D tensor of shape (H x W)
        tensor channel should be in RGB order

        rgb2bgr
        out_type, if 'np.uint8', transform outputs to uint8 type with range [0, 255]; otherwise, float type with range[0,1]
        min_max: min_max value for clamp
    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) or 2D ndarray of shape (H x W)
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)

        # min-max normalization
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:  # [B, C, H, W]
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))),
                               normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f"Only support 4D, 3D, or 2D tensor, but recevied with dimension : {n_dim}")

        if out_type == np.uint8:
            # numpy.uint8() will not round by default
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)

    if len(result) == 1:
        result = result[0]
    return result


def imfrombytes(content, img_lq_path, flag='color', float32=False):
    """ Read an image from bytes

    Args:
        content(bytes): img bytes got from files or other streams
        flag: color type of loaded images, 'color', 'grayscale' and 'unchanged'
        float32: whether to change to float32

    Returns:
        ndarray: loaded image array
    """
    NoneType = type(None)
    if isinstance(content, NoneType):
        print('cannot read', img_lq_path)
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, rgb2bgr=True, params=None, auto_mkdir=True):
    """ Write image to file
    Args:
        img(ndarray): image array to be written
        file_path: img file path
        rgb2bgr: change color mode
        params: for opencv imwrite
        auto_mkdir
    Returns:
        bool
    """
    # check rgb or grayscale
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    if len(img.shape) == 2:  # gray scale
        return cv2.imwrite(file_path, img, params)
    elif len(img.shape) == 3:  # rgb or bgr
        if rgb2bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(file_path, img, params)
    else:
        raise ValueError(f'invalid save shape {img.shape}, expect 2 or 3')


def crop_border(imgs, crop_border):
    """ Crop borders of images
    Args:
        imgs (list[ndarray]|ndarray): images with shape (h, w, c)
        crop_border: for each end of height and weight

    Returns:
        list[ndarray]: cropped images
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


def tensor_sift_warp(t_center, t_src):
    """
    Warp Tensor Based on SIFT
    :param t_center: target image
    :param t_src: adjacent frame, to be warped
    :return:
    """
    im_center = np.array(t_center * 255).astype(np.uint8)
    im_src = np.array(t_src * 255).astype(np.uint8)

    sift = cv2.xfeatures2d_SIFT.create()
    bf = cv2.BFMatcher()

    b = im_center.shape[0]
    result_im = []
    Hmats = []
    for b_idx in range(b):
        im_c = im_center[b_idx, 0]
        im_s = im_src[b_idx, 0]

        kp1, des1 = sift.detectAndCompute(im_c, None)
        kp2, des2 = sift.detectAndCompute(im_s, None)

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append(m)

        ptsA = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        Hmats.append(H)
        imgOut = cv2.warpPerspective(im_s, H, (im_c.shape[1], im_c.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        result_im.append(torch.FloatTensor(imgOut[np.newaxis, np.newaxis, :, :]))

    result_im = torch.cat(result_im, dim=0)
    return result_im / 255., Hmats


def tensor_sift_warp_from_homo(tensor, Hs, scale=4):
    im = np.array(tensor * 255)
    b = im.shape[0]
    result_im = []

    for b_idx in range(b):
        im_b = im[b_idx, 0]
        H = Hs[b_idx]
        H[:2, -1] *= scale
        imgOut = cv2.warpPerspective(im_b, H, (im_b.shape[1], im_b.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        result_im.append(torch.FloatTensor(imgOut[np.newaxis, np.newaxis, :, :]))

    result_im = torch.cat(result_im, dim=0)
    return result_im / 255.