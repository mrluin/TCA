import cv2
import random


def mod_crop(img, scale):
    """ Mod crop images, used during testing

    Args:
        img (ndarray): input images
        scale (int): scale factor

    Returns:
        ndarray: cropped images
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f"Wrong img ndim: {img.ndim}.")
    return img


def unpaired_random_crop(img_hqs, img_lqs, hq_patch_size, scale, hq_path):
    """ unPaired random crop
    It crops lists of lq and gt images with corresponding locations
    Args:
        img_hqs: (list[ndarray]|ndarray): high-quality images
        img_lqs: (list[ndarray]|ndarray): low quality images
        hq_patch_size(int): hq patch size
        scale (int): scale factor
        hq_path (str): path to high-quality data
    Returns:
        list of hq images and LQ images
    """
    if not isinstance(img_hqs, list):
        img_hqs = [img_hqs]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq = img_lqs[0].shape
    h_hq, w_hq = img_hqs[0].shape

    lq_patch_size = hq_patch_size // scale

    if h_hq != h_lq * scale or w_hq != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_hq}, {w_hq}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})")
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f"LQ ({h_lq}, {w_lq}) is smaller than patch size"
                         f"({lq_patch_size}, {lq_patch_size})"
                         f"Please remove {hq_path}")

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # unpaired cropping
    top_hq = random.randint(0, h_hq - hq_patch_size)
    left_hq = random.randint(0, w_hq - hq_patch_size)
    img_hqs = [
        v[top_hq:top_hq + hq_patch_size, left_hq:left_hq + hq_patch_size, ...]
        for v in img_hqs
    ]

    if len(img_hqs) == 1:
        img_hqs = img_hqs[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_hqs, img_lqs


def paired_random_crop(img_hqs, img_lqs, hq_patch_size, scale):
    """ Paired random crop
    It crops lists of lq and gt images with corresponding locations
    Args:
        img_hqs: (list[ndarray]|ndarray): high-quality images
        img_lqs: (list[ndarray]|ndarray): low quality images
        hq_patch_size(int): hq patch size
        scale (int): scale factor
        hq_path (str): path to high-quality data
    Returns:
        list of hq images and LQ images
    """
    if not isinstance(img_hqs, list):
        img_hqs = [img_hqs]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq = img_lqs[0].shape[:2]
    h_hq, w_hq = img_hqs[0].shape[:2]

    lq_patch_size = hq_patch_size // scale

    if h_hq != h_lq * scale or w_hq != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_hq}, {w_hq}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})")
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f"LQ ({h_lq}, {w_lq}) is smaller than patch size"
                         f"({lq_patch_size}, {lq_patch_size})")

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_hq, left_hq = int(top * scale), int(left * scale)
    img_hqs = [
        v[top_hq:top_hq + hq_patch_size, left_hq:left_hq + hq_patch_size, ...]
        for v in img_hqs
    ]

    if len(img_hqs) == 1:
        img_hqs = img_hqs[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_hqs, img_lqs


def paired_random_crop_img_flow_mask(img_hqs, img_lqs, hq_patch_size, scale,
                                     img_flows=None, img_masks=None):
    """
        cropper for img, forward and backward flows, img_masks
    """
    if not isinstance(img_hqs, list):
        img_hqs = [img_hqs]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_flows, list) and img_flows is not None:
        img_flows = [img_flows]
    if not isinstance(img_masks, list) and img_masks is not None:
        img_masks = [img_masks]

    # shape
    h_lq, w_lq = img_lqs[0].shape[:2]
    h_hq, w_hq = img_hqs[0].shape[:2]

    lq_patch_size = hq_patch_size // scale

    if h_hq != h_lq * scale or w_hq != w_lq * scale:
        raise ValueError(
            f"Scale mismatches. GT ({h_hq}, {w_hq}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq})")
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f"LQ ({h_lq}, {w_lq}) is smaller than patch size"
                         f"({lq_patch_size}, {lq_patch_size})")

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop corresponding gt patch
    top_hq, left_hq = int(top * scale), int(left * scale)
    img_hqs = [
        v[top_hq:top_hq + hq_patch_size, left_hq:left_hq + hq_patch_size, ...]
        for v in img_hqs
    ]
    # crop corresponding optical flows and masks
    if img_flows is not None:
        img_flows = [
            v[top_hq:top_hq + hq_patch_size, left_hq:left_hq + hq_patch_size, ...]
            for v in img_flows
        ]
    if img_masks is not None:
        img_masks = [
            v[top_hq:top_hq + hq_patch_size, left_hq:left_hq + hq_patch_size, ...]
            for v in img_masks
        ]

    if len(img_hqs) == 1:
        img_hqs = img_hqs[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if img_flows is not None:
        if len(img_flows) == 1:
            img_flows = img_flows[0]
    if img_masks is not None:
        if len(img_masks) == 1:
            img_masks = img_masks[0]

    if img_flows is not None and img_masks is not None:
        return img_hqs, img_lqs, img_flows, img_masks
    elif img_flows is not None and img_masks is None:
        return img_hqs, img_lqs, img_flows
    elif img_flows is None and img_masks is None:
        return img_hqs, img_lqs


def augment(imgs, hflips=True, rotation=True, flows=None, masks=None, return_status=False):
    """ Augment: hflips and rotate

    use vertical flip and transpose for rotation implementation

    Args:
        imgs (list[ndarray] | ndarray): input images, if ndarray, it will be transformed to a list
        hflips:
        rotation:
        flows (list[ndarray]): flows to be augmented
        return_status: return the status of flip and rotation
    Returns:
        list[ndarray] | ndarray: augmented images and flows
    """
    hflips = hflips and random.random() < 0.5
    vflips = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        # print(img.shape)
        if hflips:
            cv2.flip(img, 1, img)
        if vflips:
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflips:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflips:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    def _augment_mask(mask):
        if hflips:
            cv2.flip(mask, 1, mask)
        if vflips:
            cv2.flip(mask, 0, mask)
        if rot90:
            mask = mask.transpose(1, 0)
        return mask

    if not isinstance(imgs, list):
        imgs = [imgs]

    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if masks is not None:
        if not isinstance(masks, list):
            masks = [masks]
        masks = [_augment_mask(mask) for mask in masks]
        if len(masks) == 1:
            masks = masks[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]

    if flows is not None and masks is not None:
        return imgs, flows, masks
    elif flows is not None and masks is None:
        return imgs, flows
    else:
        if return_status:
            return imgs, hflips, vflips, rot90
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """ Rotate image
    Args:
        img
        angle: positive value means counter-clockwise rotation
        center (tuple[int]): rotation center
        scale (float): isotropic scale factor
    """
    h, w = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img




