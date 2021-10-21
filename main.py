'''
Main entry point for model training and evaluation. 
See samples in scripts on how to train models from scratch.

Copyright 2021 Rowel Atienza
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.misc import get_device, get_args, AverageMeter
import classifier
import configs.configs as configs
import dataset.cifar as cifar
import dataset.imagenet as imagenet
#import torch
#import numpy as np
#import random


def build_train_agmax(args, run, all_top1):
    values = classifier.build_train(args, run, all_top1)
    if not args.train:
        return
    top1, top5, model = values
    if top1 > args.best_top1:
        args.best_top1 = top1
        args.best_top5 = top5
        args.best_model = model

    return args, top1, top5

        
def main():
    args = get_args()

    if args.config is not None:
        print("%s configuration:" % args.config)
        args.agmax = configs.is_agmax(args.config)
        print("\tagmax:", args.agmax)

        if args.lr is None:
            args.lr = configs.get_lr(args.config)
        print("\tlr:", args.lr)

        if args.epochs is None:
            args.epochs = configs.get_epochs(args.config)
        print("\tepochs:", args.epochs)

        if args.weight_decay is None:
            args.weight_decay = configs.get_weight_decay(args.config)
        print("\tweight_decay:", args.weight_decay)

        if args.batch_size is None:
            args.batch_size = configs.get_batch_size(args.config)
        print("\tbatch_size:", args.batch_size)

        args.nesterov = configs.is_nesterov(args.config)
        print("\tnesterov:", args.nesterov)

        if args.init_backbone is None:
            args.init_backbone = configs.get_init_backbone(args.config)
        print("\tinit backbone:", args.init_backbone)

        if args.init_extractor is None:
            args.init_extractor = configs.get_init_extractor(args.config)
        print("\tinit extractor:", args.init_extractor)

        if args.weights_std is None:
            args.weights_std = configs.get_weights_std(args.config)
        print("\tweights_std:", args.weights_std)

        if args.dropout is None:
            args.dropout = configs.get_backbone_dropout(args.config)
        else:
            args.config = configs.set_backbone_dropout(args.config, args.dropout)
        print("\tdropout:", args.dropout)

        args.cutout = configs.is_cutout(args.config)
        print("\tcutout:", args.cutout)

        args.cutmix = configs.is_cutmix(args.config)
        print("\tcutmix:", args.cutmix)

        args.mixup = configs.is_mixup(args.config)
        print("\tmixup:", args.mixup)

        args.auto_augment =  configs.is_auto_augment(args.config)
        print("\tauto_augment:", args.auto_augment)

        if args.no_basic_augment is None:
            args.no_basic_augment =  configs.has_no_basic_augment(args.config)
        print("\tno_basic_augment:", args.no_basic_augment)

        args.feature_extractor = configs.get_backbone_name(args.config)
        print("\tbackbone:", args.feature_extractor)


    args.backbone_config = configs.get_backbone_config_by(args.feature_extractor)
    args.backbone_config['dropout'] = args.dropout

    if args.n_classes is None:
        if args.dataset == "cifar10":
            args.n_classes = cifar.SiameseCIFAR10.n_classes
            args.backbone_config['channels'] = cifar.SiameseCIFAR10.n_channels
        elif args.dataset == "cifar100":
            args.n_classes = cifar.SiameseCIFAR100.n_classes
            args.backbone_config['channels'] = cifar.SiameseCIFAR100.n_channels
        elif args.dataset == "imagenet":
            args.n_classes = imagenet.SiameseImageNet.n_classes
            args.backbone_config['channels'] = imagenet.SiameseImageNet.n_channels
        elif args.dataset == "speech_commands":
            import dataset.speech_commands_dataset as speech
            args.n_classes = speech.SpeechCommandsDataset.n_classes
            args.backbone_config['channels'] = speech.SpeechCommandsDataset.n_channels
        else:
            ValueError("Invalid number of classes")
            exit(0)

    print("Training on %s dataset with %d classes" % (args.dataset, args.n_classes))
    #print("Using seed: %d " % args.seed)
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)

    all_top1 = AverageMeter()
    all_top5 = AverageMeter()
    for run in range(1, args.n_runs + 1):
        values = build_train_agmax(args, run, all_top1)
        if args.train:
            args, top1, top5 = values
            all_top1.update(top1)
            all_top5.update(top5)
        else:
            exit(0)

    print("Top 1 Avg, Min, Max: ", all_top1.avg, all_top1.min, all_top1.max)
    print("Top 5 Avg, Min, Max: ", all_top5.avg, all_top5.min, all_top5.max)
    info = "Dataset %s, Best Top 1 %0.2f%%, Best Model %s, Avg Top 1: %0.2f%%, "
    info += "Min Top 1: %0.2f%%, Max Top 1: %0.2f%%"
    print(info
          % (args.dataset, 
             args.best_top1, 
             args.best_model,
             all_top1.avg,
             all_top1.min,
             all_top1.max))

if __name__ == '__main__':
    main()
