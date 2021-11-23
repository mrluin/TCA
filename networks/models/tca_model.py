import torch
import importlib
import copy
import os.path as osp

from tqdm import tqdm
from collections import OrderedDict
from networks.archs import define_network
from networks.models.base_model import BaseModel
from utils import get_root_logger, get_dist_info
from utils.img_utils import tensor2img, imwrite
from networks.ops.sync_batchnorm import convert_model


# loss_module = importlib.import_module('losses')
# metric_module = importlib.import_module('metrics')
eps = 1e-8


class TCAInferenceModel(BaseModel):
    def __init__(self, opt):
        super(TCAInferenceModel, self).__init__(opt)

        self.opt = opt
        self.logger = get_root_logger()
        self.network_g = define_network(copy.deepcopy(opt['network_g']))
        self.network_g = convert_model(self.network_g)
        self.print_network(self.network_g)

        load_path_g = self.opt['path'].get('pretrain_network_g', None)

        if load_path_g is not None:
            self.load_network(self.network_g, load_path_g, self.opt['path'].get('strict_load_g', True), 'network_g')
        else:
            raise ValueError('load_path_g cannot be None in inference mode.')

        self.network_g = self.model_to_device(self.network_g)

    def feed_data(self, data):
        if data.get('lq', None) is not None:
            self.lq_seq = data['lq'].to(self.device)  # input lr

    def test(self):
        self.network_g.eval()
        with torch.no_grad():
            self.output = self.network_g(self.lq_seq)
        self.network_g.train()

    def inference(self, dataloader):
        dataset_name = dataloader.dataset.opt['name']

        # dist setting
        rank, world_size = get_dist_info()

        if rank == 0:
            pbar = tqdm(total=len(dataloader.dataset), unit='clip')

        for idx in range(rank, len(dataloader.dataset), world_size):

            val_data = dataloader.dataset[idx]  # orderly get one folder/clip
            val_data['lq'].unsqueeze_(0)

            video_idx = val_data['key']  # folder/clip

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()  # list, [c, h, w], of lr, hr, and prediction, type: tensor

            sr_imgs = tensor2img(visuals['result_seq'], rgb2bgr=False, min_max=(0, 1))  # numpy  rgb

            del self.lq_seq
            del self.output

            torch.cuda.empty_cache()

            if self.opt['val']['suffix']:
                save_img_name = osp.join(
                    self.opt['path']['visualization'], dataset_name,
                    video_idx, 'sr_output', ('{idx:08d}_' + f'{self.opt["val"]["suffix"]}.png'))
            else:
                save_img_name = osp.join(
                    self.opt['path']['visualization'], dataset_name,
                    video_idx, 'sr_output', '{idx:08d}.png')

            for sr_img_idx, sr_img in zip(val_data['neighbor_list'], sr_imgs):
                imwrite(sr_img, save_img_name.format(idx=sr_img_idx))

            pbar.update(1)
            pbar.set_description(f"Test {video_idx}")
        pbar.close()

    def get_current_visuals(self):
        t = self.lq_seq.shape[1]
        lq = self.lq_seq.detach().cpu().squeeze(0)
        result = self.output.detach().cpu().squeeze(0)
        return OrderedDict([
            ('lq_seq', [lq[i] for i in range(t)]),
            ('result_seq', [result[i] for i in range(t)]),
        ])
