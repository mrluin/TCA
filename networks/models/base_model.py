import os.path as osp
import torch
import torch.nn as nn
import torch.nn.init as init
import math

import wandb

from utils.dist_utils import master_only
from torch.nn.parallel import DataParallel, DistributedDataParallel
from networks.lr_scheduler import MultiStepRestartLR, CosineAnnealingRestartLR
from collections import OrderedDict
from copy import deepcopy
from utils import get_root_logger


class BaseModel:
    """ Base model

    __all__ = [
        feed_data
        optimize_parameters
        get_current_visuals
        save
        validation
        get_current_log
        model_to_device
        setup_schedulers
        get_bare_model
        print_network
        _set_lr
        _get_init_lr
        update_learning_rate
        get_current_learning_rate
        save_network
        _print_different_keys_loading
        load_network
        save_training_state
        resume_training
        reduce_loss_dict
    ]

    """
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.logger = get_root_logger()

        self.log_dict = None
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, *args, **kwargs):
        pass

    def optimize_parameters(self, *args, **kwargs):
        pass

    def get_current_visuals(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    """
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        ''' validation phase
        Args:
            dataloader: validation dataloader
            current_iter: current iteration
            tb_logger: tensorboard logger
            save_img: whether to save imgs
        '''
        # TODO nondist_validation is not implemented, equals to dist_validation in video_base_model
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

        # dist_validation and nondist_validation are implemented in sonfile
    """

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """ Model to device
            DistributedDataParallel or DataParallel
            if dist, DistributedDataParallel mode
            elif dist is False, num_gpus > 1, DataParallel mode
            else: single GPU mode
        Args:
            net (nn.Module)
        """
        net = net.cuda()  #to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)

            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters
            )
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)

        return net

    def setup_schedulers(self):
        """ Set up schedulers
        using specific scheduler wrap each optimizer
        """
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')

        # TODO add other types of lr_scheduler
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    MultiStepRestartLR(optimizer, **train_opt['scheduler'])
                )
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    CosineAnnealingRestartLR(optimizer, **train_opt['scheduler'])
                )
        else:
            raise NotImplementedError(f"Scheduler {scheduler_type} is not implemented yet")

    def get_bare_model(self, net):
        """ get bare model, especially under wrapping with DataParallel and DistributedDataParallel"""
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = (f"{net.__class__.__name__} - "
                           f"{net.module.__class__.__name__}")
        else:
            net_cls_str = f"{net.__class__.__name__}"

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        self.logger.info(f"Network: {net_cls_str}, with parameters: {net_params:,d}")

        # self.logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """ set learning rate for warmup
        lr_groups_l (list): list for lr_groups, each for an optimizer
        [lr_groups, lr_groups, ...], each lr_groups represents lrs for one optimizer
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                # set lr for each param_group in param_groups
                param_group['lr'] = lr

    def _get_init_lr(self):
        """ get initial learning rate for each param_group
        returns: initial learning rate list
        [every optimizer, every param_groups of optimizer]
        [optimizer1.param_groups1.init_lr, ... ... optimizer2.param_groups2.init_lr, ... ...]
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append(
                [v['initial_lr'] for v in optimizer.param_groups]
            )
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """ update learning rate
        Args:
            current_iter: current iteration
            warmup_iter : -1 for no warmup
        """
        if current_iter > 1:
            # TODO modify according to pytorch version
            for scheduler in self.schedulers:
                if scheduler is not None:
                    scheduler.step()

        # set up warmup learning rate
        # has warm-up phase
        # TODO modify warm_up_lr setting
        if current_iter < warmup_iter:
            # get init_lr for each param_group
            init_lr_g_l = self._get_init_lr()
            # modify warm up learning rate
            # currently only support linearly
            warm_up_lr_l = []
            for init_lr in init_lr_g_l:
                # modify each
                warm_up_lr_l.append(
                    v / warmup_iter * current_iter for v in init_lr
                )
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        learning_rate_list = []
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                learning_rate_list.append(param_group['lr'])

        return learning_rate_list

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params', best=False):
        """
        Args:
             net (nn.Module | list[nn.Module]): networks to be saved
             net_label: network saving name
             current_iter: current iteration
             param_key (str | list[str): the parameter keys to save network
        Returns:
            checkpoint on cpu, need map_location when resuming
        """
        if current_iter == -1:
            current_iter = 'latest'
        if current_iter == -2:
            current_iter = 'initial'

        # network_iter.pth
        if best:
            save_filename = f"{net_label}_{current_iter}_best.pth"
            save_path = osp.join(self.opt['path']['models'], save_filename)
        else:
            save_filename = f"{net_label}_{current_iter}.pth"
            save_path = osp.join(self.opt['path']['models'], save_filename)

        # multiple network with multiple param_keys
        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'lengths of net and param_key should be the same'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()  # convert to cpu

            save_dict[param_key_] = state_dict
        torch.save(save_dict, save_path)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """ Print keys with different name or different size when loading models

        JUST for printing

        1. Print keys with different names
        2. If strict=False, print the same key but with different tensor size
            it also ignore these keys with different sizes(not load)
        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """

        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()

        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        # print key difference
        if crt_net_keys != load_net_keys:
            # chaji,
            self.logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                self.logger.warning(f" {v}")

            self.logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                self.logger.warning(f" {v}")

        # check the size for the same keys, not strict loading, ignore param with different shape
        if not strict:
            # jiaoji,
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    self.logger.warning(
                        f"Size different, ignore [{k}]: crt_net:"
                        f"{crt_net[k].shape}; load_net: {load_net[k].shape}"
                    )
                    load_net[k + '.ignore'] = load_net.pop(k)

        # if strict, ignore loading process

    def load_network(self, net, load_path, strict=True, param_keys='params'):
        """ Load Network
        Note: network dict includes multiple network state_dict, e.g. {'NetG_low2high', 'NetG_high2low', 'NetD'}
        Args:
            load_path (str): path to network checkpoint, one path including three subnetwork state
            net (model|list of modules): Network
            strict: whether strictly loaded
            param_keys (str|list of str): the keys of parameters need to be load in checkpoint
        """
        if not isinstance(net, list):
            net = [net]
        if not isinstance(param_keys, list):
            param_keys = [param_keys]

        assert len(net) == len(param_keys), f"network list length should equal to list of param_keys" \
                                            f"but got {len(net)} and {len(param_keys)}"
        # print('pre_load')

        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)

        for subnet, key in zip(net, param_keys):
            subnet = self.get_bare_model(subnet)
            self.logger.info(f"Loading {net.__class__.__name__} model from {load_path}")
            subnet_state_dict = load_net[key]
            for k, v in deepcopy(subnet_state_dict).items():
                if k.startswith('module.'):
                    subnet_state_dict[k[7:]] = v
                    subnet_state_dict.pop(k)  # remove module.
            self._print_different_keys_loading(subnet, subnet_state_dict, strict)
            subnet.load_state_dict(subnet_state_dict, strict=strict)

        # print('after load')

    @master_only
    def save_training_state(self, epoch, current_iter):
        """ save training state, optimizers and schedulers"""
        if current_iter != -1 and current_iter != -2:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [],
                'schedulers': []
            }
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                if s is not None:
                    state['schedulers'].append(s.state_dict())

            save_filename = f"{current_iter}.state"
            save_path = osp.join(self.opt['path']['training_states'], save_filename)
            torch.save(state, save_path)

    def resume_training(self, resume_state, learning_rate=None):
        """ reload training states, including optimizer and scheduler """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'

        # learning_rate is used to adjust post training process

        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
            if learning_rate is not None:
                self.optimizers[i]['param_groups'][0]['lr'] = learning_rate
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)


    def reduce_loss_dict(self, loss_dict):
        """ Used in distributed learning
            averages losses among different GPUs
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict

    def _log_validation_metric_values(self, epoch, current_iter, dataset_name, tb_logger):
        # average all frames for each sub-folder
        # metric_results_avg is a dict:{
        #    'folder1': tensor (len(metrics)),
        #    'folder2': tensor (len(metrics))
        # }
        metric_results_avg = {
            metric: torch.mean(tensor, dim=0).cpu()
            for (metric, tensor) in self.metric_results.items()
        }
        # total_avg_results is a dict: {
        #    'metric1': float,
        #    'metric2': float
        # }
        # for each metric
        total_avg_results = {
            metric: 0 for metric in self.opt['val']['metrics'].keys()
        }

        for metric, tensor in metric_results_avg.items():
            for idx, metric in enumerate(total_avg_results.keys()):
                total_avg_results[metric] += metric_results_avg[metric][idx]

        # average among folders
        for metric in total_avg_results.keys():
            total_avg_results[metric] /= len(metric_results_avg)

        log_str = f'Validation {dataset_name}\n'
        # report total_avg_result and each folder result
        for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
            log_str += f'\t # {metric}: {value:.4f}'
            for folder, tensor in metric_results_avg.items():
                log_str += f'\t # {folder}: {tensor[metric_idx].item():.4f}'
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)

        if tb_logger:
            # result of total average and each folder
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
                for folder, tensor in metric_results_avg.items():
                    tb_logger.add_scalar(f'metrics/{metric}/{folder}',
                                         tensor[metric_idx].item(),
                                         current_iter)

        if 'debug' not in self.opt['name'] and self.opt['logger']['wandb'] is not None:
            for metric_idx, (metric, value) in enumerate(total_avg_results.items()):
                wandb.log({f"metrics/{metric}": value, 'epoch': epoch}, commit=False)
                for index, (folder, tensor) in enumerate(metric_results_avg.items()):
                    wandb.log({f"metrics/{metric}/{folder}": tensor[metric_idx].item(), 'epoch': epoch}, commit=False)

            wandb.log({})

    @torch.no_grad()
    def _module_init(self, module, init_div_groups=False):
        """ exclude module (list of module names)
        """
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                if self.opt['init_method'] == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif self.opt['init_method'] == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                else:
                    continue
            elif isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()