from pathlib import Path

import torchvision

from data_kits import BaseDataset

cached_data = {}


class Cifar10(BaseDataset):
    """ Wrapper of torchvision.datasets.CIFAR10() """
    def __init__(self,
                 opt,
                 logger,
                 split='train',
                 augmentations=None,
                 cache=True):
        super(Cifar10, self).__init__(opt)
        _ = cache

        self.logger = logger
        self.split = split
        self.augmentation = augmentations
        self.data_dir = Path(opt.data_dir) / "Cifar10"

        self._cifar10 = torchvision.datasets.CIFAR10(
            root=str(self.data_dir.parent),
            train=True if split == 'train' else False,
            transform=augmentations,
            download=True
        )
        self.dataset_length = len(self._cifar10)
        self.num_classes = 10

    def __len__(self):
        if self.split == 'train' and self.opt.train_n > 0:
            return self.opt.train_n

        if self.split == 'val' and self.opt.val_n > 0:
            return self.opt.val_n

        return self.dataset_length

    def __getitem__(self, index):
        images, targets = self._cifar10[index % self.dataset_length]

        return {
            "img": images,
            "lab": targets
        }
