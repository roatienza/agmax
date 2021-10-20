'''
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision.transforms as transforms
from . import auto_augment as augment
from PIL import Image

# mean and std fr https://github.com/pytorch/examples/blob/master/imagenet/main.py
imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

# mean and std fr https://github.com/kakaobrain/fast-autoaugment
cifar_normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                       std=[0.2023, 0.1994, 0.2010])

# mean and std fr https://github.com/uoguelph-mlrg/Cutout
svhn_normalize = transforms.Normalize(mean=[0.4309803921568628, 0.4301960784313726, 0.4462745098039216],
                                      std=[0.19647058823529412, 0.1984313725490196, 0.19921568627450978])

# fr https://github.com/kakaobrain/fast-autoaugment
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


def speech_augment(bg_noise_dir,
                   n_mels=32):
    from . import transforms_stft as stft
    from . import transforms_wav as wav
    from . import speech_commands_dataset as speech
    data_aug_transform = transforms.Compose([wav.ChangeAmplitude(),
                                             wav.ChangeSpeedAndPitchAudio(),
                                             wav.FixAudioLength(), 
                                             stft.ToSTFT(), 
                                             stft.StretchAudioOnSTFT(), 
                                             stft.TimeshiftAudioOnSTFT(), 
                                             stft.FixSTFTDimension()])
    bg_dataset = speech.BackgroundNoiseDataset(bg_noise_dir, transform=data_aug_transform)
    add_bg_noise = stft.AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = transforms.Compose([stft.ToMelSpectrogramFromSTFT(n_mels=n_mels), 
                                                 stft.DeleteSTFT(), 
                                                 wav.ToTensor('mel_spectrogram', 'input')])

    transform_train = transforms.Compose([wav.LoadAudio(),
                                          data_aug_transform,
                                          add_bg_noise,
                                          train_feature_transform])

    test_feature_transform = transforms.Compose([wav.ToMelSpectrogram(n_mels=n_mels), 
                                                     wav.ToTensor('mel_spectrogram', 'input')])

    transform_test = transforms.Compose([wav.LoadAudio(),
                                         wav.FixAudioLength(),
                                         test_feature_transform])

    return {'train' : transform_train, 'test' : transform_test}


def data_augment(dataset,
                 length,
                 cutout=False,
                 auto_augment=False,
                 rand_augment=False,
                 rand_augment_mag=5.0,
                 no_basic_augment=False,
                 bg_noise_dir=None,
                 train_imagenet_size=224,
                 test_imagenet_size=224):

    if dataset == "speech_commands":
        return speech_augment(bg_noise_dir)

    if dataset == "imagenet":
        normalize = imagenet_normalize
    elif dataset == "svhn":
        normalize = svhn_normalize
    else:
        normalize = cifar_normalize

    # SVHN on Wide ResNet has no pre-processing 
    # https://arxiv.org/pdf/1605.07146.pdf
    # dropout = 0.4 is used
    #if no_basic_augment or dataset == "svhn" or dataset == "svhn-core":
    if no_basic_augment:
        transform_train = []
        print("No basic augment")
    else:
        # imagenet baseline transform from: 
        # https://github.com/clovaai/CutMix-PyTorch
        if dataset == "imagenet":
            transform_train = [
                #augment.EfficientNetRandomCrop(input_size),
                #transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
                transforms.RandomResizedCrop(train_imagenet_size),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_train = [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
            ]
            print("RandomCrop + RandomHorizontalFlip")
    if auto_augment:
        transform_train.append(augment.AutoAugment(dataset=dataset))
        print("AutoAugment")
    elif rand_augment:
        transform_train.append(augment.RandAugment())
        print("RandAugment")

    if dataset == "imagenet":
        transform_train.extend([
            transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
            ),
            transforms.ToTensor(),
            augment.Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            normalize,
        ])
        transform_test = transforms.Compose([
            #EfficientNetCenterCrop(input_size),
            #transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.Resize(test_imagenet_size+32),
            transforms.CenterCrop(test_imagenet_size),
            transforms.ToTensor(),
            normalize,
        ])
        print(f"Train ImageNet Size={train_imagenet_size}, Test ImageNet Size = {test_imagenet_size}")
    else:
        transform_train.extend([
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        print("CIFAR/SVHN Train and Test Transforms")

    # cutout comes after normalize
    # if before normalize, use CutOut_PIL
    if cutout:
        transform_train.append(augment.Cutout(length=length))
        print("CutOut: ", length)

    transform_train = transforms.Compose(transform_train)

    return {'train' : transform_train, 'test' : transform_test}


def color_jitter_transform():
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(
                    brightness=1.0,
                    contrast=1.0,
                    saturation=1.0,
            ),
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    print("ImageNet Color Jitter")

    return {'train' : transform_test, 'test' : transform_test}

def random_resized_crop_transform():
    transform_test = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.ToTensor(),
            imagenet_normalize,
        ])
    print("ImageNet RandomResizedCrop")

    return {'train' : transform_test, 'test' : transform_test}

class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))

