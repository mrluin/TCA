import cv2
import numpy as np

from metrics.metric_utils import reorder_image, to_y_channel


def calculate_psnr(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """ Calculate PSNR
    Args:
        img1 and img2 : images with range [0, 255]
        crop_border: cropped pixels in each edge of an image. These pixels are not involved in the PSNR calculation
        input_order: whether the input order is 'HWC' or 'CHW'
        test_y_channel: test on Y channel of YCbCr

    Returns:
        float: psnr result
    """
    assert img1.shape == img2.shape, f"Image shape are different : {img1.shape}, {img2.shape}"
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    """ Calculate SSIM for one channel images

    called by func 'calculate_ssim'

    Notes:
        similarity on luminance, contrast, and structure
        luminance: average grey value
        contrast: standard variance
        structure: x - \miu_x / \sigma_x

        C1, C2, C3 to avoid unstable case when denominator close to zero

        steps:
        1. resize, grayscale
        2. local SSIM better than global SSIM, add window, local SSIM -> SSIM matrix
        3. mean SSIM as final result

    Args:
        img1 (ndarray): image with range [0, 255] with order 'HWC'
        img2 (ndarray): image with range [0, 255] with order 'HWC'

    Returns:
        float: ssim results
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5) # use Gaussian kernel windown with standard variance 1.5
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_ssim(img1,
                   img2,
                   crop_border,
                   input_order='HWC',
                   test_y_channel=False):
    """ Calculate SSIM
    Ref:
    Image Quality Assessment: From error visibility to structural similarity

    For three-channel images, SSIM is calculated for each channel and then averaged

    Args:
        img1: Images with range [0, 255]
        img2: Images with range [0, 255]
        crop_border:
        input_order:
        test_y_channel
    Returns:
        ssim result
    """
    assert img1.shape == img2.shape, f"Image shapes are different: {img1.shape}, {img2.shape}"
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))

    return np.array(ssims).mean()
