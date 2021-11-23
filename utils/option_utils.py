import yaml
from collections import OrderedDict
import os.path as osp

__all__ = ['ordered_yaml', 'parse', 'dict2str']


def ordered_yaml():
    """Support OrderedDict for yaml

    Returns:
        yaml loader and dumper
    """

    # try to import Dumper and Loader
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    # define a representer
    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())
    # define a constructor
    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    # representer for converting OrderedDict to data.items()
    Dumper.add_representer(OrderedDict, dict_representer)
    # construct OrderedDict
    Loader.add_constructor(_mapping_tag, dict_constructor)

    return Loader, Dumper


def parse(opt_path, is_train=True):
    """ Parse option file
        .yml file to OrderedDict

    Args:
        opt_path (str): option file path
        is_train (str): in training or not
    Returns:
        (dict): Options
    """

    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # output opt OrderedDict

    opt['is_train'] = is_train

    # datasets
    for phase, dataset in opt['datasets'].items():
        # train, valid, or others
        # dataset, OrderedDict
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        # paths for resume state and pretrained network
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    # osp.abspath + pardir = path for ../
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        # path2root/experiments/exp_name
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        # path2root/experiments/exp_name/models
        opt['path']['models'] = osp.join(experiments_root, 'models')
        # path2root/experiments/exp_name/training_states
        opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
        opt['path']['log'] = experiments_root
        # path2root/experiments/exp_name/visualization
        opt['path']['visualization'] = osp.join(experiments_root, 'visualization')
        opt['path']['tb_logger'] = osp.join(experiments_root, 'tb_logger')

        # change options for debug mode
        if 'debug' in opt['name']:
            opt['num_gpu'] = 1
            if 'val' in opt:
                opt['val']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:
        # test phase
        # path2root/results/exp_name
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

        # path2root/experiments/exp_name/visualization, predictions
        opt['path']['visualization'] = osp.join(results_root, 'visualization')

        # add flows and warped results
        opt['path']['flows'] = osp.join(results_root, 'flows')
        opt['path']['warped'] = osp.join(results_root, 'warped')

    return opt


def parse_exp_analysis(opt_path):
    """ Parse option file
        from function parse
        parse for experiment analysis
    """

    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # output opt OrderedDict

    # datasets
    for phase, dataset in opt['datasets'].items():
        # train, valid, or others
        # dataset, OrderedDict
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    for key, val in opt['path'].items():
        # paths for resume state and pretrained network
        if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)

    # osp.abspath + pardir = path for ../
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))

    results_root = osp.join(opt['path']['root'], 'results', opt['name'])
    opt['path']['results_root'] = results_root
    opt['path']['log'] = results_root
    # path2root/experiments/exp_name/visualization
    opt['path']['predictions'] = osp.join(results_root, 'visualization')
    opt['path']['flows'] = osp.join(results_root, 'flows')
    opt['path']['warped'] = osp.join(results_root, 'warped')

    return opt


def dict2str(opt, indent_level=1):
    """ dict to string for printing options
    Args:
        opt (dict): option dict
        indent_level (int): indent level

    Returns:
        (str) Option string for printing
    """
    msg = '\n'
    # opt OrderedDict(... OrderedDict)
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg


def opt_get(opt, keys, default=None):
    """ recursively get the deepest element """
    if opt is None:
        return default
    rt = opt
    for key in keys:
        rt = rt.get(key, None)
        if rt is None:
            return default
    return rt


if __name__ == '__main__':

    """
    opt.items() return odict_items(key-value) recursively
    opt.get(key, setvalue)
    if exists key, return corresponding value
    otherwise, return setvalue
    """
    opt = parse('../options/example_opts.yml', is_train=True)

