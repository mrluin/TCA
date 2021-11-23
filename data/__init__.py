import importlib
import numpy as np
import random
import torch
import torch.utils.data

from utils.misc import scandir
from functools import partial
from utils.logger import get_root_logger
from utils.dist_utils import get_dist_info
from data.prefetch_dataloader import PrefetchDataLoader
import os.path as osp

# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
# e.g. reds_dataset, vimeo90k_dataset
data_folder = osp.dirname(osp.abspath(__file__))  # get dirname of _dataset.py files
dataset_filenames = [
    # splitext, ext: extension,
    # the difference between .split and .splitext
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')
]

# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f"data.{file_name}") for file_name in dataset_filenames
]


def create_dataset(dataset_opt):
    """
    Args:
        dataset_opt (dict) configuration for dataset
            name (str): dataset name
            type (str): dataset type
    """
    dataset_type = dataset_opt['type']

    # dynamic instantiation
    for module in _dataset_modules:
        # get dataset class
        # if has get, else return None
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f"data {dataset_type} is not found")

    # create dataset based on dataset_cls and dataset_opt(including dataset name and dataset type)
    dataset = dataset_cls(dataset_opt)

    logger = get_root_logger()
    logger.info(
        f"data {dataset.__class__.__name__} - {dataset_opt['name']} is created"
    )
    return dataset


def create_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """ Create dataloader
    Args:
        dataset (torch.utils.data.Dataset): data
        dataset_opt (dict): data options
            phase: train or val
            num_worker_per_gpu: number of workers for each gpu
            batch_size_per_gpu: training batch size for each gpu
        num_gpu:
        dist: whether using distributed training
        sampler (torch.utils.data.sampler): Data sampler
        seed (int|None): seed
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()

    if phase == 'train':
        # distributed training
        if dist:
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:
            # non-distributed training
            # multiplier, dataparallel training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier  # total batch_size = per_gpu * nb_gpu
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
        )

        # sampler and shuffle argument, has one
        if sampler is None:
            dataloader_args['shuffle'] = True

        # if seed is not None, setting random seed for each worker
        dataloader_args['worker_init_fn'] = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None

    elif phase in ['val', 'test']:
        dataloader_args = dict(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
    else:
        raise ValueError(f"Wrong dataset phase: {phase}"
                         f"Supported ones are 'train', 'val', 'test'")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')

    if prefetch_mode == 'cpu':
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f"Use {prefetch_mode} prefetch dataloader:"
                    f"num_prefetch_queue = {num_prefetch_queue}")
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args
        )
    else:
        #         # prefetch_mode = None: normal dataloader
        #         # prefetch_mode = 'cuda'
        return torch.utils.data.DataLoader(**dataloader_args)


def worker_init_fn(worker_id, num_workers, rank, seed):
    # set the worker seed to num_workers * rank + worker_id + seed
    # workers with different seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

