'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from features.extractor import BaseModule


class QNet(BaseModule):
    def __init__(self,
                 n_units,
                 n_classes):
        super(QNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2*n_classes, n_units),
            nn.ReLU(True),
            nn.Linear(n_units, n_classes),
        )


    def forward(self, zcat):
        zzt = self.model(zcat)
        return zzt


class AgMaxNet(BaseModule):
    def __init__(self, 
                 backbone,
                 n_units,
                 n_classes,
                 has_mi_qnet=True):
        super(AgMaxNet, self).__init__()
        
        self.backbone = backbone
        self.has_mi_qnet = has_mi_qnet

        if has_mi_qnet:
            self.qnet = QNet(n_units=n_units,
                             n_classes=n_classes)


    def forward(self, x, xt):
        size = x.size(0)
        xx = torch.cat((x, xt))
        zz = self.backbone(xx) 
        z = zz[0:size]
        zt = zz[size:]

        if not self.has_mi_qnet:
            return z, zt, None

        zcat = torch.cat((z, zt), dim=1)
        zzt = self.qnet(zcat)

        return z, zt, zzt


    def init_weights(self, std=0.01, init_backbone=True, init_extractor=False):
        super(AgMaxNet, self).init_weights(std=std)
        if not self.has_mi_qnet:
            return
        self.qnet.init_weights(std=std)
        if init_backbone:
            self.backbone.init_weights(std=std, init_extractor=init_extractor)


if __name__ == '__main__':
    pass
