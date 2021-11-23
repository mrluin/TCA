import numpy as np
import os
import random
import time
import torch
from os import path as osp

from .dist_utils import master_only
from .logger import get_root_logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """
    if path exists, rename it with timestamp and create a new one

    Args:
        path (str) folder path
    """
    if osp.exists(path):
        new_name = path + '_archieved_' + get_time_str()
        print(f"Path already exists. Rename it to {new_name}", flush=True)
        os.rename(path, new_name)

    os.makedirs(path, exist_ok=True)


@master_only
def make_exp_dirs(opt):
    """ Make dirs for experiments """
    path_opt = opt['path'].copy()
    # root path for training(experiments_root) and testing(results_root)
    if opt['is_train']:
        # path2root/experiments/exp_name
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))

    # other paths
    for key, path in path_opt.items():
        if ('strict_load' not in key) and ('pretrain_network' not in key) and ('resume' not in key) \
                and ('param_key' not in key):
            os.makedirs(path, exist_ok=True)


def check_resume(opt, resume_iter):
    """ check resume states and pretrain_network path
    Args:
        opt (dict): options
        resume_iter (int): resume iteration
    """
    # logger = get_root_logger(opt['logger_name'])

    logger = get_root_logger(opt['logger_name'])

    # if resume_state is True
    if opt['path']['resume_state']:
        # get all the network needed in training
        networks = [key for key in opt.keys() if key.startswith('network_')]

        # any network need to resume?
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f"pretrain_{network}") is not None:
                flag_pretrain = True

        if flag_pretrain:
            logger.warning('pretrain_network path will be ignored during resume')

        # set pretrained model path
        for network in networks:
            name = f"pretrain_{network}"  # pretrain_network_g
            basename = network.replace('network_', '')  # g
            # if specific network is not ignored
            if opt['path'].get('ignore_resume_networks') is None or \
                    (basename not in opt['path']['ignore_resume_networks']):
                #
                #
                checkpoint_name = opt['logger']['save_checkpoint_name']
                opt['path'][name] = osp.join(opt['path']['models'], f"{checkpoint_name}_{resume_iter}.pth")
                logger.info(f"Set {name} to {opt['path'][name]}")


def sizeof_fmt(size, suffix='B'):
    """ Get human readable file size
    Args:
         size (int): file size
         suffix (str): suffix
    Returns:
        str: formated file size
    """
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}{suffix}"
        size /= 1024.0

    return f"{size:3.1f} Y{suffix}"


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """ Scan the directory to find interested files
    Args:
        dir_path: path of the directory
        suffix ( str | tuple(str), optional):
        recursive: if true, recursively scan the directory
        full_path: if true, include dir_path
        ignore_subdirs (list): used in vimeo90k dataset, root / subfolders / seq / frame
    Returns:
        file path generator

    Notes:
        yield, return, generator
    """
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError("'suffix' must be a string or tuple of strings")

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():  # check file and dir
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)  # relative path

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    if full_path:
                        yield entry.path  # used for vimeo90k
                    else:
                        yield osp.relpath(entry.path, root)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def ascii_strcheck(name):
    return all(ord(c) < 128 for c in name)


if __name__ == '__main__':

    pass