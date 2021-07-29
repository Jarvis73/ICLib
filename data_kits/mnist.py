from pathlib import Path

import torchvision

from data_kits import BaseDataset

cached_data = {}


class Mnist(BaseDataset):
    """ Wrapper of torchvision.datasets.MNIST() """
    def __init__(self,
                 opt,
                 logger,
                 split='train',
                 augmentations=None,
                 cache=True):
        super(Mnist, self).__init__()

        self.opt = opt
        self.logger = logger
        self.split = split
        self.augmentation = augmentations
        self.cache = cache
        self.data_dir = Path(opt.data_dir) / "MNIST"

        self._mnist = torchvision.datasets.MNIST(
            root=str(self.data_dir.parent),
            train=True if split == 'train' else False,
            transform=augmentations,
            download=True
        )
        self.dataset_length = len(self._mnist)
        self.num_classes = 10

    def __len__(self):
        if self.split == 'train' and self.opt.train_n > 0:
            return self.opt.train_n

        if self.split == 'val' and self.opt.val_n > 0:
            return self.opt.val_n

        return self.dataset_length

    def __getitem__(self, index):
        image, target = self._mnist[index % self.dataset_length]

        return {
            "img": image,
            "lab": target
        }
