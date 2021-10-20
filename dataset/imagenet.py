import numpy as np
import torchvision.datasets as datasets
from PIL import Image

# default root is /data/imagenet. Download imagenet manually
# and store dataset in /data/imagenet
# useextract_ILSVRC.sh script to extract dataset tar files
# extract_ILSVRC.sh can be downloaded fr:
# https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
class SiameseImageNet(datasets.ImageNet):
    n_channels = 3
    n_classes = 1000
    def __init__(self,
                 root="/data/imagenet",
                 split='train',
                 transform=None,
                 siamese_transform=None):
        super(SiameseImageNet, self).__init__(root=root,
                                              split=split,
                                              transform=transform)
        self.siamese_transform = siamese_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            img1 = self.transform(sample)

        if self.siamese_transform is not None:
            img2 = self.siamese_transform(sample)

        #if self.transform is not None:
        #    sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target


    def __len__(self):
        return super(SiameseImageNet, self).__len__()
