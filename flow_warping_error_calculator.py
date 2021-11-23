import torch
import logging
import glob
import os.path as osp
import numpy as np

from tqdm import tqdm
from utils.misc import get_root_logger, get_time_str
from utils.img_utils import imfrombytes, img2tensor, tensor2img
from networks.archs.tca_arch import flow_warp
from networks.archs.pwc_arch import PWCNet

# GTs
gts_dict = {
    'reds4': './dataset/REDS4/GT',
    'vid4': './dataset/Vid4/GT',
    'spmc30': './dataset/SPMC30/GT',
    'vimeo90kt': './dataset/Vimeo90KT/GT',
    'udm10': './dataset/UDM10/GT'
}

vimeo90kt_test_meta_info = './data/meta_info/meta_info_vimeo90k_test.txt'


class FlowWarpingError:
    def __init__(self,
                 flow_model, logger, reds4_srs_path, vid4_srs_path, spmc_srs_path, vimeo90kt_srs_path,
                 udm10_srs_path, exp_name=None):
        super(FlowWarpingError, self).__init__()

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
        self.exp_name = exp_name

        # init accumulators
        self.flow_warp_error = []  # flow warping error on SR sequence

        assert self.exp_name is not None
        self.logger.info(f'### Metrics for {self.exp_name} \n')

    def metrics_accumulator_init(self):
        self.flow_warp_error = []

    def calculate_metrics_on(self, srs_path, check_list):
        with torch.no_grad():
            pbar = tqdm(total=len(check_list), unit='clip')
            for check in check_list:
                pbar.update(1)
                pbar.set_description(f'{check}')
                srs_list = sorted(glob.glob(osp.join(srs_path, check, '*')))

                for i in range(len(srs_list)):
                    if i > 0:
                        # previous
                        f = open(srs_list[i - 1], 'rb')
                        sr1 = f.read()
                        sr1 = imfrombytes(sr1, srs_list[i - 1], float32=True)
                        sr1 = img2tensor(sr1).unsqueeze(0).cuda()
                        # current
                        f = open(srs_list[i], 'rb')
                        sr2 = f.read()
                        sr2 = imfrombytes(sr2, srs_list[i], float32=True)
                        sr2 = img2tensor(sr2).unsqueeze(0).cuda()

                        backward_flow = self.flow_model(sr2, sr1)
                        w_sr1 = flow_warp(sr1, backward_flow.permute(0, 2, 3, 1))

                        w_sr1 = tensor2img(w_sr1)
                        sr2 = tensor2img(sr2)

                        warp_error = ((w_sr1 - sr2) ** 2).mean()
                        self.flow_warp_error.append(warp_error)

    def compute_metrics(self):
        #
        if self.reds4_srs_path is not None:
            self.logger.info('### REDS4  \n')
            self.calculate_metrics_on(self.reds4_srs_path, self.reds4_check_list)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.vid4_srs_path is not None:
            self.logger.info('### Vid4  \n')
            self.calculate_metrics_on(self.vid4_srs_path, self.vid4_check_list)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.spmc_srs_path is not None:
            self.logger.info('### SPMC-30 \n')
            self.calculate_metrics_on(self.spmc_srs_path, self.spmc_check_list)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.udm10_srs_path is not None:
            self.logger.info('### UDM10 \n')
            self.calculate_metrics_on(self.udm10_srs_path, self.udm10_check_list)
            self.statistic_and_log()
            self.metrics_accumulator_init()
        #
        if self.vimeo90kt_srs_path is not None:
            self.logger.info('### Vimeo90KT \n')
            self.calculate_metrics_on(self.vimeo90kt_srs_path, self.vimeo90kt_check_list)
            self.statistic_and_log()
            self.metrics_accumulator_init()

    def statistic_and_log(self):

        fwarp_error = np.asarray(self.flow_warp_error).sum() / len(self.flow_warp_error)

        self.logger.info(f"flow warping error {fwarp_error}\n")


if __name__ == '__main__':
    #
    resume_path = '/mnt/disk10T/jbl/pretrained/pwc-default'
    flow_model = PWCNet(pretrained=resume_path)
    flow_model = flow_model.cuda()
    flow_model.eval()

    exp_time = get_time_str()
    log_file = osp.join(f"./{exp_time}_tca_bi4x_flow_warping_error.txt")
    exp_name = '# tca_bi4x_flow_warping_error'
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file, logger_name='FlowWarpingError')

    #
    reds4_srs_path = './results/tca_reds_bi4x_model_reds4_inference/visualization/REDS4/'
    spmc_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/SPMC30'
    vid4_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/Vid4'
    vimeo90kt_srs_path = './results/tca_vimeo90k_bi4x_model_vid4_vimeo90kt_spmc30_inference/visualization/Vimeo90KT'
    udm10_srs_path = None

    flow_warping_error_calculator = FlowWarpingError(flow_model, logger, reds4_srs_path, vid4_srs_path, spmc_srs_path,
                                                     vimeo90kt_srs_path, udm10_srs_path, exp_name)
    flow_warping_error_calculator.compute_metrics()

    # BD
    exp_time = get_time_str()
    log_file = osp.join(f"./{exp_time}_tca_bd4x_flow_warping_error.txt")
    exp_name = '# tca_bd4x_flow_warping_error'
    logger = get_root_logger(log_level=logging.INFO, log_file=log_file, logger_name='FlowWarpingError')

    reds4_srs_path = None
    spmc_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/SPMC30'
    vid4_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/Vid4'
    vimeo90kt_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/Vimeo90KT'
    udm10_srs_path = './results/tca_vimeo90k_bd4x_model_udm10_vid4_vimeo90kt_spmc30_inference/visualization/UDM10'

    flow_warping_error_calculator = FlowWarpingError(flow_model, logger, reds4_srs_path, vid4_srs_path, spmc_srs_path,
                                                     vimeo90kt_srs_path, udm10_srs_path, exp_name)
    flow_warping_error_calculator.compute_metrics()
