'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import ConcatDataset

import os

class SingleLoader:
    def __init__(self,
                 root='./data',
                 batch_size=128, 
                 dataset=datasets.CIFAR10, 
                 transform={'train':transforms.ToTensor(), 'test':transforms.ToTensor()},
                 device=None,
                 dataset_name="cifar10",
                 shuffle_test=False,
                 corruption=None,
                 num_workers=16):
        super(SingleLoader, self).__init__()
        self.test = None
        self.train = None
        self._build(root,
                    batch_size, 
                    dataset, 
                    transform, 
                    device, 
                    dataset_name,
                    shuffle_test,
                    corruption,
                    num_workers)

    
    def _build(self,
               root,
               batch_size, 
               dataset, 
               transform, 
               device,
               dataset_name,
               shuffle_test,
               corruption,
               num_workers):
        DataLoader = torch.utils.data.DataLoader
        #workers = torch.cuda.device_count() * 4
        if "cuda" in str(device):
            print("num_workers: ", num_workers)
            kwargs = {'num_workers': num_workers, 'pin_memory': True}
        else:
            kwargs = {}

        if dataset_name == "svhn" or dataset_name == "svhn-core":
            x_train = dataset(root=root,
                              split='train',
                              download=True,
                              transform=transform['train'])

            if dataset_name == "svhn":
                x_extra = dataset(root=root,
                                  split='extra',
                                  download=True, 
                                  transform=transform['train'])
                x_train = ConcatDataset([x_train, x_extra])

            x_test = dataset(root=root,
                             split='test',
                             download=True,
                             transform=transform['test'])
        elif dataset_name == "imagenet":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            if corruption is None:
                x_test = dataset(root=root,
                                 split='val', 
                                 transform=transform['test'])
            else:
                root = os.path.join(root, corruption)
                corrupt_test = []
                for i in range(1, 6):
                    folder = os.path.join(root, str(i))
                    x_test = datasets.ImageFolder(root=folder,
                                                  transform=transform['test'])
                    corrupt_test.append(x_test)
                x_test = ConcatDataset(corrupt_test)

        elif dataset_name == "speech_commands":
            x_train = dataset(root=root,
                              split='train', 
                              transform=transform['train'])
            x_val = dataset(root=root,
                            split='valid', 
                            transform=transform['test'])
            x_test = dataset(root=root,
                             split='test', 
                             transform=transform['test'])

            self.val = DataLoader(x_val,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  **kwargs)

            #self.train = DataLoader(x_train,
            #                        shuffle=True,
            #                        batch_size=batch_size,
            #                        **kwargs)

            #self.test = DataLoader(x_test,
            #                       shuffle=False,
            #                       batch_size=batch_size,
            #                       **kwargs)
            #return
        else:
            x_train = dataset(root=root,
                              train=True,
                              download=True,
                              transform=transform['train'])

            x_test = dataset(root=root,
                             train=False,
                             download=True,
                             transform=transform['test'])

        self.train = DataLoader(x_train,
                                shuffle=True,
                                batch_size=batch_size,
                                **kwargs)

        self.test = DataLoader(x_test,
                               shuffle=shuffle_test,
                               batch_size=batch_size,
                               **kwargs)



class DoubleLoader(SingleLoader):
    def __init__(self,
                 root='./data',
                 batch_size=128, 
                 dataset=[None, None],
                 transform={'train':transforms.ToTensor(), 'test':transforms.ToTensor()},
                 device=None,
                 dataset_name="cifar10",
                 shuffle_test=False,
                 corruption=None,
                 num_workers=16):
        super(DoubleLoader, self).__init__(root=root,
                                           batch_size=batch_size, 
                                           dataset=dataset, 
                                           transform=transform,
                                           device=device,
                                           dataset_name=dataset_name,
                                           shuffle_test=shuffle_test,
                                           corruption=corruption,
                                           num_workers=num_workers)

    def _build(self,
               root,
               batch_size, 
               dataset, 
               transform, 
               device,
               dataset_name,
               shuffle_test,
               corruption,
               num_workers):
        print(self.__class__.__name__)
        DataLoader = torch.utils.data.DataLoader
        #workers = torch.cuda.device_count() * 4
        if "cuda" in str(device):
            print("num_workers: ", num_workers)
            kwargs = {'num_workers': num_workers, 'pin_memory': True}
        else:
            kwargs = {}

        if dataset_name == "svhn" or dataset_name == "svhn-core":
            x_train = dataset[0](root=root,
                                 split='train',
                                 download=True, 
                                 transform=transform['train'],
                                 siamese_transform=transform['train'])

            if dataset_name == "svhn":
                x_extra = dataset[0](root=root,
                                     split='extra',
                                     download=True, 
                                     transform=transform['train'],
                                     siamese_transform=transform['train'])
                x_train = ConcatDataset([x_train, x_extra])

            x_test = dataset[1](root=root,
                                split='test',
                                download=True,
                                transform=transform['test'])

        elif dataset_name == "imagenet":
            x_train = dataset[0](root=root,
                                 split='train', 
                                 transform=transform['train'],
                                 siamese_transform=transform['train'])
            if corruption is None:
                x_test = dataset[1](root=root,
                                    split='val', 
                                    transform=transform['test'])
            else:
                root = os.path.join(root, corruption)
                corrupt_test = []
                for i in range(1, 6):
                    folder = os.path.join(root, str(i))
                    x_test = datasets.ImageFolder(root=folder,
                                                  transform=transform['test'])
                    corrupt_test.append(x_test)
                x_test = ConcatDataset(corrupt_test)

        elif dataset_name == "speech_commands":
            x_train = dataset[0](root=root,
                                 split='train', 
                                 transform=transform['train'],
                                 siamese_transform=transform['train'])
            x_val = dataset[1](root=root,
                               split='valid', 
                               transform=transform['test'])
            x_test = dataset[1](root=root,
                                split='test', 
                                transform=transform['test'])
            self.val = DataLoader(x_val,
                                  shuffle=False,
                                  batch_size=batch_size,
                                  **kwargs)

            #from torch.utils.data.sampler import WeightedRandomSampler
            #weights = x_train.make_weights_for_balanced_classes()
            #sampler = WeightedRandomSampler(weights, len(weights))
            #                        sampler=sampler,

            #self.train = DataLoader(x_train,
            #                        shuffle=True,
            #                        batch_size=batch_size,
            #                        **kwargs)

            #self.test = DataLoader(x_test,
            #                       shuffle=False,
            #                       batch_size=batch_size,
            #                       **kwargs)

            #return
        else:
            x_train = dataset[0](root=root,
                                 train=True,
                                 download=True,
                                 transform=transform['train'],
                                 siamese_transform=transform['train'])
            x_test = dataset[1](root=root,
                                train=False,
                                download=True,
                                transform=transform['test'])


        self.train = DataLoader(x_train,
                                shuffle=True,
                                batch_size=batch_size,
                                **kwargs)

        self.test = DataLoader(x_test,
                               shuffle=shuffle_test,
                               batch_size=batch_size,
                               **kwargs)

