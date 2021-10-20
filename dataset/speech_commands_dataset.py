"""Google speech commands dataset."""
__author__ = 'Yuan Xu'

import torch
import os
import numpy as np

import librosa

from torch.utils.data import Dataset

__all__ = [ 'CLASSES', 'SpeechCommandsDataset', 'BackgroundNoiseDataset' ]


CLASSES_10 = 'unknown, silence, yes, no, up, down, left, right, on, off, stop, go'.split(', ')
#CLASSES_30 = 'unknown, silence, bed, bird, cat, dog, down, eight, five, four, go, happy, house, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero'.split(', ')
#SILENCE_INDEX = 1

CLASSES_30 = 'silence, bed, bird, cat, dog, down, eight, five, four, go, happy, house, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, wow, yes, zero'.split(', ')
SILENCE_INDEX = 0

class SpeechCommandsDataset(Dataset):
    """Google speech commands dataset. 
    """
    n_channels = 1
    n_classes = 31 + SILENCE_INDEX
    def __init__(self,
                 root, 
                 split, 
                 transform=None, 
                 classes=CLASSES_30, 
                 silence_percentage=0.05,
                 n_unknown_per_class=5):
        folder = os.path.join(root, split)
        all_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        for c in classes[SILENCE_INDEX+1:]:
            assert c in all_classes

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        for c in all_classes:
            if c not in class_to_idx:
                class_to_idx[c] = 0

        data = []
        for c in all_classes:
            d = os.path.join(folder, c)
            target = class_to_idx[c]
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append((path, target))

        self.paths = np.array(data)[:,0]
        self.unknown_key = 'unknown'
        # add silence
        target = class_to_idx['silence']
        data += [('', target)] * int(len(data) * silence_percentage)

        if split == 'train':
            # add unknown
            if SILENCE_INDEX > 0:
                target = class_to_idx[self.unknown_key]
                data += [(self.unknown_key, target)] * int(len(classes) * n_unknown_per_class)

        self.split = split
        self.classes = classes
        self.data = data
        self.transform = transform
        #print(class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path, target = self.data[index]
        if self.split == 'train' and path == self.unknown_key:
            rand_index = np.random.randint(0, len(self.paths))
            path = self.paths[rand_index]

        data = {'path': path, 'target': target}

        if self.transform is not None:
            data = self.transform(data)

        x = data['input']
        x = torch.unsqueeze(x, 0)
        return x, data['target']

    def make_weights_for_balanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(self.classes)
        count = np.zeros(nclasses)
        for item in self.data:
            count[item[1]] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx, item in enumerate(self.data):
            weight[idx] = weight_per_class[item[1]]
        return weight

class SiameseSpeechCommandsDataset(SpeechCommandsDataset):
    def __init__(self,
                 root="/data/speech/speech_commands/dataset",
                 split='train',
                 transform=None,
                 siamese_transform=None):
        super(SiameseSpeechCommandsDataset, self).__init__(root=root,
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
        path, target = self.data[index]
        if self.split == 'train' and path == self.unknown_key:
            rand_index = np.random.randint(0, len(self.paths))
            path = self.paths[rand_index]

        data = {'path': path, 'target': target}

        if self.transform is not None:
            data1 = self.transform(data)

        if self.siamese_transform is not None:
            data2 = self.siamese_transform(data)

        x1 = data1['input']
        x1 = torch.unsqueeze(x1, 0)
        x2 = data2['input']
        x2 = torch.unsqueeze(x2, 0)
        return [x1, x2], data1['target']


    def __len__(self):
        return super(SiameseSpeechCommandsDataset, self).__len__()


class BackgroundNoiseDataset(Dataset):
    """Dataset for silence / background noise."""

    def __init__(self, folder, transform=None, sample_rate=16000, sample_length=1, classes=CLASSES_30):
        audio_files = [d for d in os.listdir(folder) if os.path.isfile(os.path.join(folder, d)) and d.endswith('.wav')]
        samples = []
        for f in audio_files:
            path = os.path.join(folder, f)
            s, sr = librosa.load(path, sample_rate)
            samples.append(s)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r*c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.classes = classes
        self.transform = transform
        self.path = folder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {'samples': self.samples[index], 'sample_rate': self.sample_rate, 'target': SILENCE_INDEX, 'path': self.path}

        if self.transform is not None:
            data = self.transform(data)

        return data
