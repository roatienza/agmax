'''
Several backbone networks are supported:
    1) WideResNet
    2) ResNet50/101
    3) EfficientNet
    4) VGG
    5) LeNet
    6) RegNet

Under features folder, the different backbone models can be found.
I started using the model implementation. 
Eventually (EfficientNet and RegNet), I found it easier to load the backbone using timm modules.

Can be easily expanded to support other models. 
1) Under features folder, add the backbone class. You might need to print the 
model summary to determine the number of units before the last Linear layer.
2) Edit the get_backbone() function to support the new model.
3) Edit the config files to support your new model.

Copyright 2021 Rowel Atienza
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from features.extractor import BaseModule
from features.wide_resnet import WideResNet
from features.resnet import ResNet
from features.vgg import VGG
from features.lenet import LeNet

from features.efficientnet import EfficientNetB1, EfficientNetB0
from features.regnet import RegNetX002, RegNetY004


def get_backbone(dataset, n_classes, pool_size, feature_extractor, backbone_config):
    if "WideResNet" in feature_extractor:
        feature_extractor = WideResNet(backbone_config, feature_extractor)
    elif "EfficientNetB0" in feature_extractor:
        feature_extractor = EfficientNetB0(backbone_config, feature_extractor)
    elif "EfficientNetB1" in feature_extractor:
        feature_extractor = EfficientNetB1(backbone_config, feature_extractor)
    elif "RegNetX002" in feature_extractor:
        feature_extractor = RegNetX002(backbone_config, feature_extractor)
    elif "RegNetY004" in feature_extractor:
        feature_extractor = RegNetY004(backbone_config, feature_extractor)
    elif "ResNet" in feature_extractor:
        feature_extractor = ResNet(backbone_config, feature_extractor)
    elif "VGG" in feature_extractor:
        feature_extractor = VGG(backbone_config, feature_extractor)
    elif "LeNet" in feature_extractor:
        feature_extractor = LeNet(backbone_config, feature_extractor)
    else:
        ValueError("Unknown feature extractor network", feature_extractor)
        exit(0)

    backbone = Backbone(feature_extractor,
                        n_classes=n_classes, 
                        pool_size=pool_size)


    if backbone is None:
        ValueError("Invalid backbone")
        exit(0)

    return backbone



class Backbone(BaseModule):
    def __init__(self, feature_extractor, n_classes, pool_size):
        super(Backbone, self).__init__()
        self.feature_extractor = feature_extractor
        self.name = self.feature_extractor.name

        n_features = feature_extractor.n_features
        if pool_size == 0:
            kernel_size = 8
            self.pool = nn.AvgPool2d(kernel_size)
            self.classifier = nn.Linear(n_features, n_classes)
        elif "LeNet" in self.name:
            self.pool = nn.MaxPool2d(2)
            self.classifier = nn.Sequential(
                    nn.Linear(n_features, 120),
                    nn.ReLU(True),
                    nn.Linear(120, 84),
                    nn.ReLU(True),
                    nn.Linear(84, n_classes),
                )
        else:
            self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
            self.classifier = nn.Linear(n_features * pool_size * pool_size, n_classes)


    def forward(self, x):
        features = self.feature_extractor(x)
        if self.classifier is None:
            return features
        z = self.pool(features)
        z = torch.flatten(z, 1)
        output = self.classifier(z)
        return output


    def init_weights(self, std=0.01, init_extractor=False):
        super(Backbone, self).init_weights(std=std)
        if init_extractor:
            self.feature_extractor.init_weights(std=std)
