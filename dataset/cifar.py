
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

class SiameseCIFAR10(datasets.CIFAR10):
    n_channels = 3
    n_classes = 10
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 siamese_transform=None,
                 download=True):
        super(SiameseCIFAR10, self).__init__(root,
                                             train=train,
                                             transform=transform,
                                             download=download)
        self.siamese_transform = siamese_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        #img2 = Image.fromarray(img)
    
        # TODO: might save memory if only 1 image is transformed
        if self.transform is not None:
            img1 = self.transform(img)

        if self.siamese_transform is not None:
            img2 = self.siamese_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target

    def __len__(self):
        return super(SiameseCIFAR10, self).__len__()


class SiameseCIFAR100(datasets.CIFAR100):
    n_channels = 3
    n_classes = 100
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 siamese_transform=None,
                 download=True):
        super(SiameseCIFAR100, self).__init__(root,
                                              train=train,
                                              transform=transform,
                                              download=download)
        self.siamese_transform = siamese_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        #img2 = Image.fromarray(img)

        if self.transform is not None:
            img1 = self.transform(img)

        if self.siamese_transform is not None:
            img2 = self.siamese_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [img1, img2], target

    def __len__(self):
        return super(SiameseCIFAR100, self).__len__()
