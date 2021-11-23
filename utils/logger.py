import datetime
import logging
import time

import wandb

from .dist_utils import get_dist_info, master_only


class MessageLogger:
    """ Message logger for printing
        Need arguments related to log printing
    Args:
        opt (dict), configurations
            name: experience name
            logger: contains 'print_freq' for logger interval, 'save_checkpoint_freq', 'use_tb_logger'
            train: contains 'total_iter' for total iter
            use_tb_logger: use tensorboard logger

        start_iter: start iteration
        tb_logger (obj: `tb_logger`): tensorboard logger
    """
    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.opt = opt
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iter = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger

        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """ Logging message
        Training
        Time
        others
        tb_logger, log losses, other metrics should be added manually
        Args:
            log_vars (dict)
                epoch: epoch number
                iter: current iter
                lrs: list for learning rates
                time: iter time
                data_time: data time for each iter
        """
        # get epoch, iter, learning rates from log_vars
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        # message: [exp_name .. [epoch: , iter: , lr:( , , )]]
        message = f"[{self.exp_name[:5]} .. ][epoch:{epoch:3d}, iter:{current_iter:8d}, lr:("
        for v in lrs:
            message += f"{v:.3e}, "
        message += ")]"

        # time and estimated time
        # message [eta: {}] [time (data): {} ({})]
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')
            total_time = time.time() - self.start_time

            time_sec_avg = total_time / (current_iter - self.start_iter + 1)  # time cost per iteration
            eta_sec = time_sec_avg * (self.max_iter - current_iter - 1)  # rest time
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))      # rest time str
            message += f"[eta: {eta_str}"
            message += f"[time (data): {iter_time:.3f} ({data_time:.3f})]"

        # other items
        for k, vt in log_vars.items():
            message += f"[{k}: {vt:.4e}]"
            # tensorboard logger
            # TODO add to tb_logger, prefix, need to log other metrics
            # losses prefix, loss_
            # other metrics
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('loss_'):
                    self.tb_logger.add_scalar(f"losses/{k}", vt, current_iter)
                else:
                    self.tb_logger.add_scalar(k, vt, current_iter)

            # wandb logger in each iteration
            if 'debug' not in self.exp_name and self.opt['logger']['wandb'] is not None:
                wandb.log({f"losses/{k}": vt, 'iteration': current_iter}, commit=False)
                wandb.log({})

        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """ Now only use wandb to sync tensorboard log"""
    import wandb
    logger = logging.getLogger()

    # opt.logger.wandb: 'project' and 'resume_id'
    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')

    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f"Resume wandb logger with id={wandb_id}")
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    run = wandb.init(
        project=project,
        entity='mrluin',
        config=opt,
        tags=['reproduce', 'baseline', '1e-6'],
        name=opt['name'],
        notes='visualize the last frame of each val video',
        mode='online',
        resume=resume,
        sync_tensorboard=False,
        id=wandb_id,
        reinit=True,
    )

    logger.info(f"use wandb logger with id={wandb_id}; project={project}")
    return run


def get_root_logger(log_level=logging.INFO, log_file=None, logger_name='VideoSR-Collection'):
    """ Get the root logger
    The logger will be initialized if it has not been initialized.
    By default a StreamHandler will be added.
    If `log_file` if specified, a FileHandler will also be added
    
    Args:
        logger_name: root logger name.
        log_file: the log filename.
        log_level: root logger level.
    Returns:
        logging.Logger: the root logger
    """
    """Helps:
        logger 
        StreamHandler, return log to output streams, like sys.stdout, sys.stderr
        FileHandler, return log to disk
        NullHandler
    """
    logger = logging.getLogger(logger_name)
    # if already exist, return
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    rank, _ = get_dist_info()

    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    return logger


def get_env_info():
    import torch
    import PIL
    import sys
    import os
    import torchvision

    msg = ''
    msg += ('\nVersion Information:'
            f"\n\tPython Version {sys.version}"
            f"\n\tPillow Version {PIL.__version__}"
            f"\n\tPyTorch Version {torch.__version__}"
            f"\n\tTorchvision Version {torchvision.__version__}"
            f"\n\tcuDNN Version {torch.backends.cudnn.version()}"
            f"\n\tCUDA available {torch.cuda.is_available()}"
            f"\n\tCUDA GPU numbers {torch.cuda.device_count()}"
            f"\n\tCUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else None}")

    return msg
