import glob
import torch
import numpy as np

from os import path as osp
from torch.utils import data as data
from PIL import Image

from data.data_utils import generate_frame_indices, read_img_seq
from utils import get_root_logger, scandir
from utils.img_utils import imfrombytes, img2tensor


class VideoInferenceDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoInferenceDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']

        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.imgs_lq = {}

        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]  # 000 011 015 020
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))  # root_to_data / *

        if opt['name'].lower() in ['vid4', 'reds4', 'spmc30', 'udm10']:
            for index, subfolder_lq, in enumerate(subfolders_lq):
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))

                max_idx = len(img_paths_lq)

                self.data_info['lq_path'].extend(img_paths_lq)

                self.data_info['folder'].extend([subfolder_name] * max_idx)  # give each frame folder name
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')

                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = []
                    for path in img_paths_lq:
                        img = Image.open(path)
                        img = np.array(img).astype(np.uint8)
                        self.imgs_lq[subfolder_name].append(img)
                    self.imgs_lq[subfolder_name] = img2tensor(self.imgs_lq[subfolder_name])
                    self.imgs_lq[subfolder_name] = torch.stack(self.imgs_lq[subfolder_name], dim=0)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')  # frame_num, total number of frame
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)

        return {
            'lq': imgs_lq,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': lq_path,
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


class VideoRecurrentInferenceDataset(VideoInferenceDataset):
    def __init__(self, opt):
        super(VideoRecurrentInferenceDataset, self).__init__(opt)
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            img_paths_lq = self.imgs_lq[folder]

            imgs_lq = []
            for path in img_paths_lq:
                f = open(path, 'rb')
                img_lq = f.read()
                img_lq = imfrombytes(img_lq, path, float32=True)
                imgs_lq.append(img_lq)
            imgs_lq = img2tensor(imgs_lq)
            imgs_lq = torch.stack(imgs_lq, dim=0)

        neighbor_list = list(range(len(imgs_lq)))

        rt_result = {
            'lq': imgs_lq,
            'key': folder,
            'neighbor_list': neighbor_list,
        }

        return rt_result

    def __len__(self):
        return len(self.folders)


class Vimeo90KInferenceRecurrentDataset(data.Dataset):
    def __init__(self, opt):
        super(Vimeo90KInferenceRecurrentDataset, self).__init__()

        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented')
        self.lq_root = opt['dataroot_lq']
        self.data_info = {
            'lq_path': [],
            'folder': [],
            'idx': [],
            'border': []
        }
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.imgs_lq = {}

        # get information from meta_info_file for Vimeo90K_T
        if opt.get('meta_info_file', None) is not None:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
            for idx, subfolder in enumerate(subfolders):
                lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in self.neighbor_list]
                self.data_info['lq_path'].append(lq_paths)
                self.data_info['folder'].append(f'{subfolder}')  # 000001/000266
                self.data_info['border'].append(0)
        else:
            raise NotImplementedError('Should provide meta_info file when using vimeo90k testing')

    def __getitem__(self, index):

        lq_path = self.data_info['lq_path'][index]
        imgs_lq = read_img_seq(lq_path)

        return {
            'lq': imgs_lq,  # (1, t, c, h, w)
            'key': self.data_info['folder'][index],  # idx/subfolder_name, e.g. 000001/0266
            'neighbor_list': self.neighbor_list
        }

    def __len__(self):
        return len(self.data_info['hq_path'])
