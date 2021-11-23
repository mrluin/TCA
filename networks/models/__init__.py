import importlib
import os.path as osp

from utils import get_root_logger, scandir


# network/models for adding additional method for specific network
# network/models/archs for defining network architectures

# automatically scan and import model modules
# scan all the files under 'network' folder and collect files ending with '_model.py'

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    # all .py files end with '_model' in the same dir
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')
]
# import all the detected model modules
_model_modules = [
    importlib.import_module(f"networks.models.{file_name}") for file_name in model_filenames
]


def create_model(opt):
    """ Create model
    Args:
        opt (dict): configuration, contains
            model_type
    """
    module_type = opt['model_type']

    for module in _model_modules:
        # from 'module' py file import module_type class
        model_cls = getattr(module, module_type, None)
        if model_cls is not None:
            break

    if model_cls is None:
        raise ValueError(f"Model {module_type} is not found")

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f"Model [{model.__class__.__name__}] is created")

    return model
