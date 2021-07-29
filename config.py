import random
from pathlib import Path

import numpy as np
import torch
from sacred import SETTINGS
from sacred import Experiment
from sacred.config.custom_containers import ReadOnlyDict
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from utils.loggers import get_global_logger

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "datasets"
DEFAULT_LOG_DIR = PROJECT_DIR / "runs"

SETTINGS.DISCOVER_SOURCES = "sys"
SETTINGS.DISCOVER_DEPENDENCIES = "sys"


def add_observers(ex,
                  config,
                  fileStorage=False,
                  MongoDB=True,
                  db_name="DEFAULT"):
    if fileStorage:
        observer_file = FileStorageObserver(config["logdir"])
        ex.observers.append(observer_file)

    if MongoDB:
        observer_mongo = MongoObserver(
            url=f"{config['mongo_host']}:{config['mongo_port']}",
            db_name=db_name)
        ex.observers.append(observer_mongo)


def assert_in_error(name, lst, value):
    raise ValueError(
        f"`{name}` must be selected from [{'|'.join(lst)}], got '{value}'")


def setup_config(ex: Experiment):
    # Track outputs
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.config
    def configurations():
        root = str(PROJECT_DIR)         # root path
        model_name = 'simclr'           # select model. [simclr]
        backbone = 'resnet50'           # backbone model. [resnet50|resnet101]
        logdir = ''                     # experiment checkpoint directory
        mongo_host = 'localhost'        # mongodb host
        mongo_port = 7010               # mongodb port
        seed = 1234                     # random seed

        # training
        lr = 0.01
        bs = 4                          # training batch size
        train_epochs = 120              # Number of epochs to train for.
        train_steps = 0                 # Number of steps to train for. If provided, overrides train_epochs.
        eval_steps = 0                  # Number of steps to eval for. If not provided, evals over entire dataset.
        weight_decay = 1e-6             # Amount of weight decay to use
        checkpoint_epochs = 1           # Number of epochs between checkpoints/summaries.
        checkpoint_steps = 0            # Number of steps between checkpoints/summaries. If provided, overrides checkpoint_epochs

        # testing/evaluation
        eval_bs = 4                     # Batch size for eval.

        #model
        bn = 'bn'                       # normalization layer [bn]
        bn_decay = 0.9                  # Batch norm decay parameter
        ckpt_id = -1                    # resume model by experiment id (only resume the best model)
        ckpt = ''                       # resume model by a specific path

        #data
        dataset = 'mnist'               # select dataset [mnist]
        data_dir = 'datasets'           # Directory where dataset is stored.
        no_droplast = False
        no_shuffle = False              # do not use shuffle
        no_aug = False                  # do not use data augmentation
        train_n = 0                     # If > 0, then #samples per epoch is set to <= train_n
        val_n = 0                       # If > 0, then #samples is set to <= val_n
        ignore_index = 255              # ignore index of labels, used in loss functions
        cache_dataset = False           # Whether to cache the entire dataset in memory. If the dataset is ImageNet, this is a very bad idea, but for smaller datasets it can improve performance

        num_workers = 6
        image_size = 32                 # Input image size.
        color_jitter_strength = 0.5     # The strength of color jittering.

        # simclr parameters
        if model_name == 'simclr':
            proj_out_dim = 64               # Number of head projection dimension.
            temperature = 0.2               # Temperature parameter for contrastive loss.
            lineareval_while_pretraining = False # Whether to finetune supervised head while pretraining
            hidden_norm = True              # Temperature parameter for contrastive loss.
            proj_head_mode = 'nonlinear'    # How the head projection is done. [none|linear|nonlinear]
            num_proj_layers = 3             # Number of non-linear head layers.
            ft_proj_selector = 0            # Which layer of the projection head to use during fine-tuning. 0 means no projection head, and -1 means the final layer.
            fine_tune_after_block = -1      # finetune model after which block
            zero_init_logits_layer = False  # If True, zero initialize layers after avg_pool for supervised learning

        # solver
        lrp = "poly"                    # Learning rate policy [custom_step|period_step|plateau|cosine|poly]
        if lrp == "custom_step":
            lr_boundaries = []          # (custom_step) Use the specified lr at the given boundaries
        if lrp == "period_step":
            lr_step = 999999999         # (period_step) Decay the base learning rate at a fixed step
        if lrp in ["custom_step", "period_step", "plateau"]:
            lr_rate = 0.1               # (period_step, plateau) Learning rate decay rate
        if lrp in ["plateau", "cosine", "poly"]:
            lr_end = 0.                 # (plateau, cosine, poly) The minimal end learning rate
        if lrp == "plateau":
            lr_patience = 30            # (plateau) Learning rate patience for decay
            lr_min_delta = 1e-4         # (plateau) Minimum delta to indicate improvement
            cool_down = 0               # (plateau)
            monitor = "val_loss"        # (plateau) Quantity to be monitored [val_loss|loss]
        if lrp == "poly":
            power = 0.9                 # (poly)
        if lrp == 'warmup_cosine':
            lr_scaling = 'sqrt'         # How to scale the learning rate as a function of batch size. [linear|sqrt]
            warmup_epochs = 10          # Number of epochs of warmup

        optimizer = "sgd"               # Optimizer for training [sgd|adam]
        if optimizer == "adam":
            adam_beta1 = 0.9            # (adam) Parameter
            adam_beta2 = 0.999          # (adam) Parameter
            adam_epsilon = 1e-8         # (adam) Parameter
        if optimizer == "sgd":
            sgd_momentum = 0.9          # (momentum) Parameter
            sgd_nesterov = False        # (momentum) Parameter

    @ex.named_config
    def pretrain_cifar10():
        dataset = 'cifar10'
        backbone = 'resnet18'
        cache_dataset = True            # not used. Cached already.

        lr = 0.2
        lrp = 'warmup_cosine'
        bs = 256
        train_epochs = 400
        sgd_nesterov = True

    @ex.named_config
    def train_then_eval_cifar10():
        dataset = 'cifar10'
        backbone = 'resnet18'
        cache_dataset = True            # not used. Cached already.

        lr = 0.1
        lrp = 'cosine'
        bs = 512
        eval_bs = 512
        train_epochs = 100
        sgd_nesterov = True
        fine_tune_after_block = 4
        zero_init_logits_layer = True

    @ex.config_hook
    def config_hook(config, command_name, logger):
        add_observers(ex, config, db_name=ex.path)
        ex.logger = get_global_logger(name=ex.path)

        # Type check
        assert_list = ["mean_rgb", "multi_grid", "lr_boundaries"]
        for x in assert_list:
            if x not in config:
                continue
            if not isinstance(config[x], (list, tuple)):
                raise TypeError(f"`{x}` must be a list or tuple, got "
                                f"{type(config[x])}")

        return config

    # add missed source files by sacred
    for source_file in (PROJECT_DIR / "data_kits").glob("*.py"):
        ex.add_source_file(str(source_file))


class MapConfig(ReadOnlyDict):
    """
    A wrapper for dict. This wrapper allow users to access dict value by `dot`
    operation. For example, you can access `cfg["split"]` by `cfg.split`, which
    makes the code more clear. Notice that the result object is a
    sacred.config.custom_containers.ReadOnlyDict, which is a read-only dict for
    preserving the configuration.

    Parameters
    ----------
    obj: ReadOnlyDict
        Configuration dict.
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = MapConfig(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super(MapConfig, self).__init__(new_dict, **kwargs)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_rundir(opt, _run):
    if _run._id is not None:
        return str(Path(opt.logdir or DEFAULT_LOG_DIR) / str(_run._id))
    elif opt.logdir:
        return opt.logdir
    else:
        return str(Path(DEFAULT_LOG_DIR) / "None")


def setup_runner(ex, _run, _config):
    opt = MapConfig(_config)    # Access configurations by attribute
    set_seed(opt.seed)
    logger = get_global_logger(name=ex.path)
    logger.info(f"RUNDIR: {get_rundir(opt, _run)}")
    return opt, logger
