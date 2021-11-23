import torch
import glob
import logging
import os.path as osp
import numpy as np
import lpips

from tqdm import tqdm
from utils.misc import set_random_seed, get_time_str
from utils.logger import get_root_logger
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from utils.img_utils import tensor2img, imfrombytes, img2tensor
from networks.archs.tca_arch import flow_warp
from networks.archs.pwc_arch import PWCNet

gts_dict = {
    'reds4': './dataset/REDS4/GT',
    'vid4': './dataset/Vid4/GT',
    'spmc30': './dataset/SPMC30/GT',
    'vimeo90kt': './dataset/Vimeo90KT/GT',
    'udm10': './dataset/UDM10/GT'
}

vimeo90kt_test_meta_info = './data/meta_info/meta_info_vimeo90k_test.txt'


class Metrics_calculator:
    def __init__(self,
                 flow_model, lpips_model, logger, reds4_srs_path, vid4_srs_path, spmc_srs_path, vimeo90kt_srs_path,
                 udm10_srs_path, exp_name=None):
        super(Metrics_calculator, self).__init__()

        # paths and check list
        self.reds4_check_list = ['000', '011', '015', '020']
        self.vid4_check_list = ['calendar', 'city', 'foliage', 'walk']
        # generate spmc check list
        self.spmc_check_list = []
        spmc_subdirs = glob.glob(osp.join(gts_dict['spmc30'], '*'))
        for subdir in spmc_subdirs:
            _, dirname = osp.split(subdir)
            self.spmc_check_list.append(dirname)
        # generate vimeo90kt check list
        self.vimeo90kt_check_list = []
        with open(vimeo90kt_test_meta_info, 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]  # 000/000
            for subfolder in subfolders:
                self.vimeo90kt_check_list.append(subfolder)
        # generate udm10 check list
        self.udm10_check_list = []
        udm_subdirs = glob.glob(osp.join(gts_dict['udm10'], '*'))
        for subdir in udm_subdirs:
            _, dirname = osp.split(subdir)
            self.udm10_check_list.append(dirname)

        self.reds4_srs_path = reds4_srs_path
        self.vid4_srs_path = vid4_srs_path
        self.spmc_srs_path = spmc_srs_path
        self.vimeo90kt_srs_path = vimeo90kt_srs_path
        self.udm10_srs_path = udm10_srs_path

        self.logger = logger

        self.flow_model = flow_model
        self.loss_fn_vgg = lpips_model
        self.exp_name = exp_name

        # init accumulators
        self.psnr = []
        self.ssim = []
        self.lpips = []

        assert self.exp_name is not None
        self.logger.info(f'### Metrics for {self.exp_name} \n')

    def metrics_accumulator_init(self):
        self.psnr = []
        self.ssim = []
        self.lpips = []

    def calculate_metrics_on(self, check_type, srs_path, check_list, crop_border=None, y_channel=True):
        with torch.no_grad():

            pbar = tqdm(total=len(check_list), unit='clip')
            for check in check_list:
                pbar.update(1)
                pbar.set_description(f'{check}')
                srs_list = sorted(glob.glob(osp.join(srs_path, check, '*')))
                hrs_list = sorted(glob.glob(osp.join(gts_dict[check_type], check, '*.png')))

                for i in range(len(srs_list)):
                    #
                    f = open(srs_list[i], 'rb')
                    sr = f.read()
                    sr = imfrombytes(sr, srs_list[i], float32=True)
                    sr = img2tensor(sr).unsqueeze(0).cuda()
                    #
                    f = open(hrs_list[i], 'rb')
                    hr = f.read()
                    hr = imfrombytes(hr, hrs_list[i], float32=True)
                    hr = img2tensor(hr).unsqueeze(0).cuda()
                    #
                    flow = self.flow_model(sr, hr)
                    a_hr = flow_warp(hr, flow.permute(0, 2, 3, 1))

                    # crop
                    if crop_border is not None:
                        c_sr = sr[:, :, crop_border:-crop_border, crop_border:-crop_border]
                        ca_hr = a_hr[:, :, crop_border:-crop_border, crop_border:-crop_border]
                    else:
                        c_sr = sr
                        ca_hr = a_hr

                    psnr = calculate_psnr(tensor2img(c_sr), tensor2img(ca_hr), crop_border=0, test_y_channel=y_channel)
                    ssim = calculate_ssim(tensor2img(c_sr), tensor2img(ca_hr), crop_border=0, test_y_channel=y_channel)
                    lpips = self.loss_fn_vgg(c_sr, ca_hr)

                    self.psnr.append(psnr)
                    self.ssim.append(ssim)
                    self.lpips.append(lpips.cpu().numpy())

    def compute_metrics(self):
        #
        if self.reds4_srs_path is not None:
            self.logger.info('### REDS4  \n')
            self.calculate_metrics_on('reds4', self.reds4_srs_path, self.reds4_check_list, 12, False)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.vid4_srs_path is not None:
            self.logger.info('### Vid4  \n')
            self.calculate_metrics_on('vid4', self.vid4_srs_path, self.vid4_check_list, 12, True)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.spmc_srs_path is not None:
            self.logger.info('### SPMC-30 \n')
            self.calculate_metrics_on('spmc30', self.spmc_srs_path, self.spmc_check_list, 12, True)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.udm10_srs_path is not None:
            self.logger.info('### UDM10 \n')
            self.calculate_metrics_on('udm10', self.udm10_srs_path, self.udm10_check_list, 12, True)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.vimeo90kt_srs_path is not None:
            self.logger.info('### Vimeo90KT \n')
            self.calculate_metrics_on('vimeo90kt', self.vimeo90kt_srs_path, self.vimeo90kt_check_list, 12, True)
            self.statistic_and_log()
            self.metrics_accumulator_init()

    def statistic_and_log(self):
        psnr = np.asarray(self.psnr).mean()
        ssim = np.asarray(self.ssim).mean()
        lpips = np.asarray(self.lpips).mean()

        self.logger.info(f"PSNR  {psnr}\n"
                         f"SSIM  {ssim}\n"
                         f"LPIPS {lpips}\n")


if __name__ == '__main__':
    #
    pwc_checkpoint_path = './checkpoints/pwc-default'
    flow_model = PWCNet(pretrained=pwc_checkpoint_path)
    flow_model = flow_model.cuda()
    flow_model.eval()
    #
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    #
    exp_time = get_time_str()
    log_file = osp.join(f"./{exp_time}-tca_bi4x_metrics_statistics.txt")
    exp_name = 'tca_bi4x_metrics_statistics'
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)
    #
    set_random_seed(1)
    #
    reds4_srs_path = './results/tca_reds_bi4x_model_reds4_inference/visualization/REDS4/'
    spmc_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/SPMC30'
    vid4_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/Vid4'
    vimeo90kt_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/Vimeo90KT'
    udm10_srs_path = None
    metrics_calc = Metrics_calculator(flow_model, loss_fn_vgg, logger, reds4_srs_path, vid4_srs_path,
                                      spmc_srs_path, vimeo90kt_srs_path, udm10_srs_path, exp_name)
    metrics_calc.compute_metrics()

    # BD
    exp_time = get_time_str()
    log_file = osp.join(f"./{exp_time}-tca_bd4x_metrics_statistics.txt")
    exp_name = 'tca_bd4x_metrics_statistics'
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file)
    #
    set_random_seed(1)
    #
    reds4_srs_path = None
    spmc_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/SPMC30'
    vid4_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/Vid4'
    vimeo90kt_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/Vimeo90KT'
    udm10_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/UDM10'
    metrics_calc = Metrics_calculator(flow_model, loss_fn_vgg, logger, reds4_srs_path, vid4_srs_path,
                                      spmc_srs_path, vimeo90kt_srs_path, udm10_srs_path, exp_name)
    metrics_calc.compute_metrics()
