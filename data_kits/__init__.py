import importlib
import torch.utils.data
from torchvision import transforms as T
from data_kits.base_dataset import BaseDataset
from utils.loggers import C as CC


def find_dataset_by_name(name):
    """Import the module "data_kits/<name>.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data_kits." + name
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    for _name, cls in datasetlib.__dict__.items():
        if _name.lower() == name.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            f"In {dataset_filename}.py, there should be a subclass of "
            f"BaseDataset with class name that matches {name} in lowercase.")

    return dataset


class PairGenerator(object):
    def __init__(self, transform, n_views=2):
        self.transform = transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.n_views)]


def get_augmentations(opt, mode='train', contrast=False):
    if mode == 'train':
        if contrast:
            s = opt.color_jitter_strength
            color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            transform = T.Compose([T.RandomResizedCrop(size=opt.image_size),
                                   T.RandomHorizontalFlip(),
                                   T.RandomApply([color_jitter], p=0.8),
                                   T.RandomGrayscale(p=0.2),
                                   T.ToTensor()])
            return PairGenerator(transform)
        else:
            transform = T.Compose([T.RandomResizedCrop(size=opt.image_size),
                                   T.RandomHorizontalFlip(),
                                   T.ToTensor()])
            return transform
    else:
        return T.Compose([T.Resize(size=opt.image_size),
                          T.ToTensor()])


class Dataset(object):
    def __init__(self, opt, logger, splits=('train', 'val'), augmentations=None, contrast=False):
        self.opt = opt
        self.logger = logger

        ds_cls = find_dataset_by_name(opt.dataset)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if 'train' in splits:
            preprocess_fn = None if opt.no_aug \
                else (augmentations or get_augmentations(opt, 'train', contrast))
            self.train_dataset = ds_cls(opt, logger, augmentations=preprocess_fn, split='train')
            self.logger.info("Dataset " +
                             CC.c(f"{self.train_dataset.__class__.__name__}", CC.BOLD) +
                             " for training was created" +
                             " (pair)" * contrast)

            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.bs,
                shuffle=not opt.no_shuffle,
                num_workers=int(opt.num_workers),
                drop_last=not opt.no_droplast,
                pin_memory=True,
            )
            logger.info(f"Found {len(self.train_dataset)} training examples")

        if 'val' in splits:
            preprocess_fn = get_augmentations(opt, 'val')
            self.val_dataset = ds_cls(opt, logger, augmentations=preprocess_fn, split='val')
            self.logger.info("PairDataset " +
                             CC.c(f"{self.val_dataset.__class__.__name__}", CC.BOLD) +
                             " for validating was created")

            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=opt.eval_bs,
                shuffle=False,
                num_workers=int(opt.num_workers),
                drop_last=False,
                pin_memory=True,
            )
            logger.info(f"Found {len(self.val_dataset)} validating examples")

        if 'test' in splits:
            preprocess_fn = get_augmentations(opt, 'val')
            self.test_dataset = ds_cls(opt, logger, augmentations=preprocess_fn, split='test')
            self.logger.info("PairDataset " +
                             CC.c(f"{self.test_dataset.__class__.__name__}", CC.BOLD) +
                             " for testing was created")

            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=opt.eval_bs,
                shuffle=False,
                num_workers=int(opt.num_workers),
                drop_last=False,
                pin_memory=True,
            )
            logger.info(f"Found {len(self.test_dataset)} testing examples")
