
from . import resnet
from . import wide_resnet
from . import regnet
from . import efficientnet

standard_weights_std = 0.01
agmax_weights_std = 0.2
#wrn282_agmax_weights_std = 0.02
#lenet_agmax_weights_std = 0.1
#vgg_agmax_weights_std = 0.2

train = {**resnet.train, **wide_resnet.train, **regnet.train, **efficientnet.train}
parameters = {**resnet.parameters, **wide_resnet.parameters, **regnet.parameters, **efficientnet.parameters}
backbone_config = {**resnet.backbone_config, **wide_resnet.backbone_config, **regnet.backbone_config, **efficientnet.backbone_config}

def get_lr(config):
    return train[config]["parameters"]["lr"]

def get_epochs(config):
    return train[config]["parameters"]["epochs"]

def get_weight_decay(config):
    return train[config]["parameters"]["weight_decay"]

def get_batch_size(config):
    return train[config]["parameters"]["batch_size"]

def is_nesterov(config):
    return train[config]["parameters"]["nesterov"]

def get_init_backbone(config):
    return train[config]["parameters"]["init_backbone"]

def get_init_extractor(config):
    return train[config]["parameters"]["init_extractor"]

def get_weights_std(config):
    return train[config]["weights_std"]

def is_agmax(config):
    return train[config]["agmax"]

def is_cutout(config):
    return train[config]["cutout"]

def is_auto_augment(config):
    return train[config]["auto_augment"]

def has_no_basic_augment(config):
    return train[config]["no_basic_augment"]

def is_cutmix(config):
    return train[config]["cutmix"]

def is_mixup(config):
    return train[config]["mixup"]

def get_backbone_name(config):
    return train[config]["backbone"]

def get_backbone_depth(config):
    return train[config]["backbone_config"]["depth"]

def get_backbone_width(config):
    return train[config]["backbone_config"]["width"]

def get_backbone_dropout(config):
    return train[config]["backbone_config"]["dropout"]

def set_backbone_dropout(config, dropout):
    train[config]["backbone_config"]["dropout"] = dropout
    return config

def get_backbone_config(config):
    return train[config]["backbone_config"]

def get_backbone_config_by(feature_extractor):
    return backbone_config[feature_extractor]


if __name__ == '__main__':
    configs = ['WideResNet28-10-standard', 'WideResNet28-10-cutout', 'WideResNet28-10-cutout-auto_augment', 'WideResNet28-10-no_basic_augment']
    for config in configs:
        print(config)
        print("\tagmax:", is_agmax(config))
        print("\tlr:", get_lr(config))
        print("\tepochs:", get_epochs(config))
        print("\tbatch_size:", get_batch_size(config))
        print("\tnesterov:", is_nesterov(config))
        print("\tweight_decay:", get_weight_decay(config))
        print("\tweights_std:", get_weights_std(config))
        print("\tcutout:", is_cutout(config))
        print("\tauto_augment:", is_auto_augment(config))
        print("\tno_basic_augment:", has_no_basic_augment(config))
        print("\tbackbone:", get_backbone_name(config))
        print("\tbackbone_depth:", get_backbone_depth(config))
        print("\tbackbone_width:", get_backbone_width(config))
        print("\tbackbone_dropout:", get_backbone_dropout(config))
