import logging
import torch
import argparse
import random
import os.path as osp

from data import create_dataset, create_dataloader
from networks.models import create_model
from utils import get_root_logger, get_env_info, get_time_str, make_exp_dirs, init_dist, get_dist_info, set_random_seed
from utils.option_utils import dict2str, parse


def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, default=None, help='path to option YAML file')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed setting
    # launcher set to none, close distributed learning
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    # if torch.distributed is not available, rank=0, world_size=1
    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
        set_random_seed(seed + opt['rank'])

    return opt


def main():
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(log_level=logging.INFO,
                             log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    test_loaders = []
    for phase, dataset_opt in opt['datasets'].items():
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']
        )
        logger.info(f"number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f"Testing {test_set_name} ... ")
        model.inference(
            test_loader,
        )


if __name__ == '__main__':
    main()
