'''

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import StepLRScheduler, CosineLRScheduler

import datetime
import numpy as np
import os
import math
import models
import backbones
import dataset.cifar as cifar
import dataset.imagenet as imagenet
import utils.misc as misc
from loss import agmax_loss, cross_entropy_loss, cross_entropy
from utils.misc import get_device, get_args, AverageMeter, fgsm_attack, count_parameters
from utils.ui import progress_bar
from utils.metrics import accuracy
from dataset.transform import data_augment, color_jitter_transform, random_resized_crop_transform
from dataset.auto_augment import cutmix, mixup
from dataloaders import SingleLoader, DoubleLoader


class Classifier:
    def __init__(self,
                 args, 
                 backbone, 
                 dataloader,
                 device=get_device()):
        super(Classifier, self).__init__()
        self.args = args
        self.backbone = backbone
        self.dataloader = dataloader

        self.device = device
        self.best_top1 = 0
        self.best_top5 = 0
        self.best_epoch = 0 
        self.milestones = [30, 60, 80]

            
        self._build_model()


    def _build_model(self):
        self.model = self.backbone
        if self.args.summary:
            print(self.model)
            param_count = count_parameters(self.model) / 1e6
            print("Model parameters: %0.1fM" % param_count)
        self._build()


    def get_model_name(self):
        if self.args.agmax:
            model_name = self.args.dataset + "-agmax-"
            if self.args.agmax_mse:
                model_name += "mse-"
            elif self.args.agmax_kl:
                model_name += "kl-"
            elif self.args.agmax_ce:
                model_name += "ce-"
            else:
                model_name += "mi-"
        else:
            model_name = self.args.dataset + "-standard-"

        model_name += self.backbone.name + "-"
        if self.args.cutout:
            model_name += "cutout-"
        if self.args.cutmix:
            model_name += "cutmix-"
        if self.args.mixup:
            model_name += "mixup-"
        if self.args.auto_augment:
            model_name += "auto_augment-"
        if self.args.rand_augment:
            model_name += "rand_augment-"
        if self.args.no_basic_augment:
            model_name += "no_basic_augment-"

        return model_name


    def _log_loss(self, epoch, ce, agreement, dl):
        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        filename = model_name + "train-loss.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        if epoch == 1:
            logs = ["Epoch,CE,Entropy,L1"]
            logs.append("%d,%f,%f,%f" % (epoch, ce, agreement, dl))
        else:
            logs = ["%d,%f,%f,%f" % (epoch, ce, agreement, dl)]

        for log in logs:
            filename.write(log)
            filename.write("\n")
        filename.close()


    def _log_acc(self, epoch, top1, top5, is_val=False, eps=0., val_name=None):
        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        if is_val:
            filename = model_name + "val-acc.log"
        elif eps > 0:
            filename = model_name + "fgsm-acc.log"
        elif val_name is not None:
            filename = model_name + val_name + "-acc.log"
        else:
            filename = model_name + "test-acc.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        if epoch == 1:
            logs = ["---------%s--------%s---------" % \
                    (model_name, datetime.datetime.now())]
            logs.append("Epoch,Top1,Top5")
            logs.append("%d,%f,%f" % (epoch, top1, top5))
        else:
            if eps > 0:
                logs = ["FGSM Attack: Epsilon %0.2f, Top-1 %f, Top-5 %f" % (eps, top1, top5)]
            else:
                logs = ["%d,%f,%f" % (epoch, top1, top5)]

        for log in logs:
            filename.write(log)
            filename.write("\n")
        filename.close()



    def _log(self, top1=None, top5=None, verbose=True):

        folder = self.args.logs_dir
        os.makedirs(folder, exist_ok=True)
        model_name = self.get_model_name()
        if top1 is None:
            filename = model_name + "start.log"
        else:
            filename = model_name + "end.log"
        path = os.path.join(folder, filename)
        filename = open(path, "a+")
        logs = ["---------%s--------%s---------" % \
                (model_name, datetime.datetime.now())]
        logs.append("Device: %s" % self.device)
        logs.append("Dataset: %s" % self.args.dataset)
        logs.append("Number of classes: %d" % self.args.n_classes)
        mi_agreement = not self.args.agmax_mse and not self.args.agmax_kl and not self.args.agmax_ce and self.args.agmax
        if mi_agreement:
            logs.append("Agreement by MI")
            logs.append("Q Network 1st Dense layer # units: %d" % self.args.n_units)
            logs.append("Q Network weights std: %f" % self.args.weights_std)
            logs.append("Init backbone: %s" % self.args.init_backbone)
            logs.append("Init extractor: %s" % self.args.init_extractor)
        elif self.args.agmax and self.args.agmax_mse:
            logs.append("Agreement by MSE")
        elif self.args.agmax and self.args.agmax_kl:
            logs.append("Agreement by KL")
        elif self.args.agmax and self.args.agmax_ce:
            logs.append("Agreement by CE")

        logs.append("Backbone: %s" % self.backbone.name)
        logs.append("Batch size: %d" % self.args.batch_size)
        if self.args.adam:
            logs.append("Adam optimizer")
        if self.args.rmsprop:
            logs.append("RMSprop optimizer")
        else:
            logs.append("SGD optimizer momentum: %f" % self.args.momentum)
            logs.append("Nesterov: %s" % self.args.nesterov)
        if self.args.multisteplr:
            logs.append("Multistep learning rate")
            logs.append("Milestones: %s" % self.milestones)
        elif self.args.steplr:
            logs.append("Step learning rate")
            logs.append("Decay epochs: %0.2f" % self.args.decay_epochs)
            logs.append("Decay rate: %0.2f" % self.args.decay_rate)
            logs.append("Warmup lr: %f" % self.args.warmup_lr)
            logs.append("Warmup epochs: %f" % self.args.warmup_epochs)
        elif self.args.cosinelr:
            logs.append("Cosine learning rate decay with warmup")
            logs.append("Warmup lr: %f" % self.args.warmup_lr)
            logs.append("Warmup epochs: %f" % self.args.warmup_epochs)
            logs.append("Cycle limit: %d" % self.args.cycle_limit)
        elif self.args.plateau:
            logs.append("Reduce on plataeu")
        else:
            logs.append("Cosine learning rate decay")
        logs.append("Weight decay: %f" % self.args.weight_decay)
        logs.append("LR: %f" % self.args.lr)
        logs.append("Epochs: %d" % self.args.epochs)
        logs.append("Dropout: %f" % self.args.dropout)
        logs.append("Rand Augment: %s, Auto Augment: %s, No Basic Augment: %s, CutOut: %s, CutMix: %s, MixUp: %s, AgMax: %s, KL: %s, MSE: %s, CE: %s" \
                % (
                   self.args.rand_augment,
                   self.args.auto_augment,
                   self.args.no_basic_augment, 
                   self.args.cutout,
                   self.args.cutmix,
                   self.args.mixup,
                   self.args.agmax,
                   self.args.agmax_kl,
                   self.args.agmax_mse,
                   self.args.agmax_ce))
        if self.args.rand_augment:
            logs.append("RandAugment Mag: %s" % self.args.rand_augment_mag)
        if self.args.cutmix:
            logs.append("CutMix Beta: %s" % self.args.beta)
            logs.append("CutMix Probability: %s" % self.args.cutmix_prob)
        if self.args.mixup:
            logs.append("MixUp Alpha: %s" % self.args.alpha)
        if mi_agreement:
            logs.append("DL Weight: %f" % self.args.dl_weight)
            logs.append("DL: %s" % self.args.dl)
        if top1 is not None:
            logs.append("Best top 1 accuracy: %f" % top1)
        if top5 is not None:
            logs.append("Best top 5 accuracy: %f" % top5)

        logs.append("---------%s--------%s---------" % \
                    (model_name, datetime.datetime.now()))

        for log in logs:
            filename.write(log)
            filename.write("\n")
            if verbose:
                print(log)
        filename.close()


    def assign_lr_scheduler(self, last_epoch=-1):
        if self.args.multisteplr:
            if self.args.epochs <= 30:
                self.milestones = [10, 20, 30]
            elif self.args.epochs <= 60:
                self.milestones = [15, 30, 40]
            elif self.args.epochs <= 120:
                self.milestones = [30, 60, 80]
            else:
                self.milestones = [75, 150, 225]
            self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.1, last_epoch=last_epoch)
        elif self.args.steplr:
            self.scheduler = StepLRScheduler(self.optimizer,
                                             decay_t=self.args.decay_epochs,
                                             decay_rate=self.args.decay_rate,
                                             warmup_lr_init=self.args.warmup_lr,
                                             warmup_t=self.args.warmup_epochs)

        elif self.args.cosinelr:
            self.scheduler = CosineLRScheduler(self.optimizer,
                                               t_initial=self.args.epochs,
                                               #decay_t=self.args.decay_epochs,
                                               #decay_rate=self.args.decay_rate,
                                               cycle_limit=self.args.cycle_limit,
                                               warmup_prefix=True,
                                               warmup_lr_init=self.args.warmup_lr,
                                               warmup_t=self.args.warmup_epochs)

        elif self.args.plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.args.epochs, last_epoch=last_epoch)


    def _build(self, init_weights=False):
        self.model = self.model.to(self.device)

        # init Q net of AgMax
        if init_weights and self.args.weights_std > 0:
            self.model.init_weights(std=self.args.weights_std,
                                    init_backbone=self.args.init_backbone,
                                    init_extractor=self.args.init_extractor)

        if "cuda" in str(self.device):
            self.model = torch.nn.DataParallel(self.model)
            print("Data parallel:", self.device)

        cudnn.benchmark = True

        if self.args.adam:
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.args.lr,
                                        weight_decay=self.args.weight_decay)
        elif self.args.rmsprop:
            # decay (alpha or smoothing) 0.9, momentum 0.9, 
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.args.lr,
                                           momentum=0.9,
                                           eps=0.001,
                                           alpha=0.9,
                                           weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.args.lr,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay,
                                       nesterov=self.args.nesterov)
        self.assign_lr_scheduler()
        self._log()

        #lr = 0.5 * initial_lr * (
        #        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))

    def prepare_train(self, run, best_top1, best_top5, best_model, all_top1, epoch):
        best_model = "None" if best_model is None else best_model
        info = "\nRun %d(%d), " 
        info += "Epoch %d(%d), PID %d, "
        info += "Dataset: %s, Best Top 1: %0.2f%%, Best Top 5: %0.2f%% Best Model: %s"
        if run > 1:
            info += ", Avg Top 1: %0.2f%%, Min Top 1: %0.2f%%, Max Top 1: %0.2f%%"
            print(info % (run, self.args.n_runs, epoch, self.args.epochs,
                          os.getpid(), self.args.dataset, best_top1, best_top5, best_model,
                          all_top1.avg, all_top1.min, all_top1.max))
        else:
            print(info % (run, self.args.n_runs, epoch, self.args.epochs,
                          os.getpid(), self.args.dataset, best_top1, best_top5, best_model))
        self.model.train()


    def train(self, run, best_top1, best_top5, best_model, all_top1, epoch, label_smoothing=0):
        self.prepare_train(run, best_top1, best_top5, best_model, all_top1, epoch)
        if self.args.steplr or self.args.cosinelr:
            lr = self.scheduler.get_epoch_values(epoch)
        else:
            lr = [self.optimizer.param_groups[0]['lr']] if self.args.plateau else self.scheduler.get_last_lr()
        lr = lr[0]
        correct = 0
        total = 0
        losses = AverageMeter()

        if label_smoothing > 0:
            ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        else:
            ce_loss = nn.CrossEntropyLoss()        
        for i, data in enumerate(self.dataloader.train):
            image, target = data
            x = image.to(self.device)
            target = target.to(self.device)

            is_cutmix = self.args.cutmix and (np.random.rand(1)[0] < self.args.cutmix_prob)
            is_mixup = self.args.mixup
            if is_cutmix:
                x, target_a, target_b, lam  = cutmix(x, 
                                                     target=target,
                                                     beta=self.args.beta,
                                                     device=self.device)
            elif is_mixup:
                x, target_a, target_b, lam = mixup(x,
                                                   target=target,
                                                   alpha=self.args.alpha,
                                                   device=self.device)


            y = self.model(x)

            self.optimizer.zero_grad()
            if is_cutmix or is_mixup:
                loss = ce_loss(y, target_a) * lam + ce_loss(y, target_b) * (1. - lam)
            else:
                loss = ce_loss(y, target)
            loss.backward()
            self.optimizer.step()

            losses.update(loss.float().mean().item())
                
            _, predicted = y.max(1)

            total += target.size(0)
            if is_mixup:
                correct += (lam * predicted.eq(target_a).sum().item()
                           + (1 - lam) * predicted.eq(target_b).sum().item())
            else:
                correct += predicted.eq(target).sum().item()
            acc = correct * 100. / total

            if label_smoothing > 0:
                ce_name = "Smooth CE"
            else:
                ce_name = "CE"
            progress_bar(i,
                         len(self.dataloader.train), 
                         '%s: %.4f | Top 1 Acc: %0.2f%% | LR: %.2e'
                         % (ce_name, losses.avg, acc, lr))
        
        return losses.avg
        

    def eval(self, epoch=0, is_val=False, val_name=None):
        self.backbone.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        extra = " with AgMax" if self.args.agmax else ""
        if is_val:
            loader = self.dataloader.val
            dset = "val"
        else:
            loader = self.dataloader.test
            dset = "test"
        with torch.no_grad():
            for i, data in enumerate(loader):
                x, target = data
                x = x.to(self.device)
                target = target.to(self.device)

                y = self.backbone(x)
                acc1, acc5 = accuracy(y, target, (1, 5))
                top1.update(acc1[0], x.size(0))
                top5.update(acc5[0], x.size(0))

                progress_bar(i,
                             len(self.dataloader.test), 
                             '%s%s %s %s accuracy: Top 1: %0.2f%%, Top 5: %0.2f%%'
                             % (self.backbone.name, extra, self.args.dataset, dset, top1.avg, top5.avg))
                
            if self.best_top1 > 0 and not is_val:
                info = "Epoch %d top 1 accuracy: %0.2f%%"
                info += ", Old best top 1 accuracy: %0.2f%% at epoch %d"
                data = (epoch, top1.avg, self.best_top1, self.best_epoch)
                print(info % data)

            if top1.avg > self.best_top1 and not is_val:
                self.best_top1 = top1.avg.float().item()
                self.best_top5 = top5.avg.float().item()
                self.best_epoch = epoch
                info = "New best top1: %0.2f%%, top5: %0.2f%%"
                print(info % (self.best_top1, self.best_top5))
                folder = self.args.weights_dir
                os.makedirs(folder, exist_ok=True)
                self.best_model = self.get_model_name()
                self.best_model += str(round(self.best_top1,2)) 
                if self.args.agmax and not (self.args.agmax_mse or self.args.agmax_kl or self.args.agmax_ce):
                    self.best_model += "-mlp-" + str(self.args.n_units) +  ".pth"
                else:
                    self.best_model += ".pth"
                path = os.path.join(folder, self.best_model)
                self.save_checkpoint(epoch, path=path, is_best=True)

            if self.args.save:
                self.save_checkpoint(epoch)
            
            self._log_acc(epoch, top1.avg.float().item(), top5.avg.float().item(), is_val=is_val, val_name=val_name)

        return self.best_top1, self.best_top5, self.best_model


    def eval_robustness(self, epsilon, epoch=0, is_val=False):
        self.backbone.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()
        extra = " with AgMax" if self.args.agmax else ""

        if is_val:
            loader = self.dataloader.val
            dset = "val"
        else:
            loader = self.dataloader.test
            dset = "test"

        # make sure batch size is 1
        for i, data in enumerate(loader):
            x, target = data
            x = x.to(self.device)

            # Set requires_grad attribute of tensor. Important for Attack
            x.requires_grad = True

            target = target.to(self.device)

            # Forward pass the data through the model
            y = self.backbone(x)
            init_pred = y.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                acc1, acc5 = accuracy(y, target, (1, 5))
                top1.update(acc1[0], x.size(0))
                top5.update(acc5[0], x.size(0))
                continue

            # Calculate the loss
            loss = nn.CrossEntropyLoss()(y, target)

            # Zero all existing gradients
            self.backbone.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = x.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(x, epsilon, data_grad)
            # Re-classify the perturbed image
            y = self.backbone(perturbed_data)

            acc1, acc5 = accuracy(y, target, (1, 5))
            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            progress_bar(i,
                         len(self.dataloader.test), 
                         '%s%s %s %s accuracy: Eps: %0.2f, Top 1: %0.4f%%, Top 5: %0.4f%%'
                         % (self.backbone.name, extra, self.args.dataset, dset, epsilon, top1.avg, top5.avg))
                

        self._log_acc(epoch, top1.avg.float().item(), top5.avg.float().item(), is_val=is_val, eps=epsilon)

        return self.best_top1, self.best_top5, self.best_model


    def save_checkpoint(self, epoch, path=None, is_best=False):
        if not is_best:
            folder = self.args.checkpoints_dir
            os.makedirs(folder, exist_ok=True)
            filename = self.get_model_name() + "epoch-" + str(epoch) + ".pth"
            path = os.path.join(folder, filename)

        print("Saving checkpoint ... ", path)
        checkpoint = {'epoch': epoch,
                      'best_top1': self.best_top1,
                      'best_top5': self.best_top5,
                      'best_epoch': self.best_epoch,
                      'best_model': self.best_model,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict' : self.optimizer.state_dict(),
                      'scheduler_state_dict' : self.scheduler.state_dict(),
                     }
        torch.save(checkpoint, path)



class AgMaxClassifier(Classifier):
    def __init__(self,
                 args, 
                 backbone, 
                 dataloader,
                 device=get_device()):
        super(AgMaxClassifier, self).__init__(args,
                                              backbone,
                                              dataloader,
                                              device=device)

    def _build_model(self):
        if self.args.n_units == 0:
            factor = int(math.log10(self.args.n_classes))
            factor = 2 ** factor if factor > 1 else 1
            self.args.n_units = 128 * factor
            if self.args.n_classes >= 1000:
                # x2 for 2048, x4 for 4096
                #self.args.n_units *= 2
                self.args.n_units *= 4

        self.not_mi = self.args.agmax_mse or self.args.agmax_kl or self.args.agmax_ce
        if self.not_mi:
            has_mi_qnet = False
        else:
            has_mi_qnet = True
        self.model = models.AgMaxNet(backbone=self.backbone,
                                     n_units=self.args.n_units,
                                     n_classes=self.args.n_classes,
                                     has_mi_qnet=has_mi_qnet).to(self.device)
        if self.args.summary:
            print(self.model)
            param_count = count_parameters(self.model) / 1e6
            print("Model parameters: %0.1fM" % param_count)
            param_count = count_parameters(self.model.backbone) / 1e6
            print("Backbone parameters: %0.1fM" % param_count)
            if self.model.has_mi_qnet:
                param_count = count_parameters(self.model.qnet) / 1e6
                print("QNet parameters: %0.1fM" % param_count)


        init_weights = True if (self.args.init_backbone and self.args.init_extractor) else False
        #if self.args.init_backbone and self.args.init_extractor:
        #    init_weights = False

        self._build(init_weights=init_weights)


    def train(self, run, best_top1, best_top5, best_model, all_top1, epoch, label_smoothing=0):
        self.prepare_train(run, best_top1, best_top5, best_model, all_top1, epoch)
        if self.args.steplr or self.args.cosinelr:
            lr = self.scheduler.get_epoch_values(epoch)
        else:
            lr = [self.optimizer.param_groups[0]['lr']] if self.args.plateau else self.scheduler.get_last_lr()
        lr = lr[0]
        correct = 0
        total = 0
        agreement_losses = AverageMeter()
        dl_losses = AverageMeter()
        ce_losses = AverageMeter()

        ce_loss = nn.CrossEntropyLoss()

        for i, data in enumerate(self.dataloader.train):
            image, target = data
            x = image[0].to(self.device)
            xt = image[1].to(self.device)
            target = target.to(self.device)

            is_cutmix = self.args.cutmix and (np.random.rand(1)[0] < self.args.cutmix_prob)
            is_mixup = self.args.mixup
            if is_cutmix:
                x, target_a, target_b, lam  = cutmix(x, 
                                                     target=target,
                                                     beta=self.args.beta,
                                                     device=self.device)
                xt, target_at, target_bt, lamt = cutmix(xt, 
                                                        target=target,
                                                        beta=self.args.beta,
                                                        device=self.device)
            elif is_mixup:
                x, target_a, target_b, lam = mixup(x,
                                                   target=target,
                                                   alpha=self.args.alpha,
                                                   device=self.device)
                xt, target_at, target_bt, lamt = mixup(xt,
                                                       target=target,
                                                       alpha=self.args.alpha,
                                                       device=self.device)
                
            y = self.model(x, xt)
            z, zt, _ = y
            self.optimizer.zero_grad()
            if is_cutmix or is_mixup:
                ce  = ce_loss(z,  target_a ) * lam  + ce_loss(z,  target_b ) * (1. - lam )
                ce += ce_loss(zt, target_at) * lamt + ce_loss(zt, target_bt) * (1. - lamt)
                ce *= 0.5
            else:
                ce = cross_entropy_loss(z, zt, target, label_smoothing=label_smoothing)

            if self.args.agmax_mse:
                agreement_loss = nn.MSELoss()(z, zt)
            elif self.args.agmax_kl:
                Pz  = nn.LogSoftmax(dim=1)(z)
                Pzt = nn.LogSoftmax(dim=1)(zt)
                agreement_loss = nn.KLDivLoss(log_target=True, reduction='batchmean')(Pz, Pzt)
            elif self.args.agmax_ce:
                agreement_loss = cross_entropy(z, zt)
            else:
                agreement_loss, dl = agmax_loss(y, target, self.args.dl_weight)
                #loss = agreement_loss + dl + ce

            loss = agreement_loss + ce
            if not self.not_mi:
                loss += dl
            loss.backward()
            self.optimizer.step()

            if self.args.steplr or self.args.cosinelr:
                fractional_epoch = epoch - 1 + i/(1.0*len(self.dataloader.train))
                self.scheduler.step(fractional_epoch)
                lr = self.scheduler.get_epoch_values(fractional_epoch)
                lr = lr[0]
            else:
                fractional_epoch = epoch - 1


            ce_losses.update(ce.float().mean().item())
            agreement_losses.update(agreement_loss.float().mean().item())
            if not self.not_mi:
                dl_losses.update(dl.float().mean().item())

            _, predicted = z.max(1)
            total += target.size(0)
            if is_mixup:
                correct += (lam * predicted.eq(target_a).sum().item()
                           + (1 - lam) * predicted.eq(target_b).sum().item())
            else:
                correct += predicted.eq(target).sum().item()
            acc = correct * 100. / total

            if self.not_mi:
                progress_bar(i,
                             len(self.dataloader.train), 
                             'AG: %.3f | CE: %.3f | Top1 Acc: %0.2f%% | LR: %.4e | Ep: %.1f'
                            % (agreement_losses.avg,
                                ce_losses.avg, 
                                acc,
                                lr,
                                fractional_epoch))
            else:
                progress_bar(i,
                             len(self.dataloader.train), 
                             'AG: %.3f | DL: %.3f | CE: %.3f | Top1 Acc: %0.2f%% | LR: %.4e | Div: %s | DL W: %.1f'
                            % (agreement_losses.avg,
                                dl_losses.avg, 
                                ce_losses.avg, 
                                acc,
                                lr, 
                                self.args.dl, 
                                self.args.dl_weight))

        self._log_loss(epoch, ce_losses.avg, agreement_losses.avg, dl_losses.avg)
        return ce_losses.avg


def build_train(args, run, all_top1):
    folder = args.weights_dir
    os.makedirs(folder, exist_ok=True)
    length = 16
    net = Classifier
    root = './data'
    bg_noise_dir = None
    if args.agmax:
        net = AgMaxClassifier
        if args.dataset == "cifar10":
            print("CIFAR10 agmax")
            train_dataset = cifar.SiameseCIFAR10
            test_dataset = datasets.CIFAR10
        elif args.dataset == "cifar100":
            print("CIFAR100 agmax")
            train_dataset = cifar.SiameseCIFAR100
            test_dataset = datasets.CIFAR100
        elif args.dataset == "imagenet":
            print("ImageNet agmax")
            train_dataset = imagenet.SiameseImageNet
            test_dataset = datasets.ImageNet
            root = args.imagenet_dir
            # fr CutMix https://arxiv.org/pdf/1905.04899.pdf
            length = 112
        elif args.dataset == "speech_commands":
            import dataset.speech_commands_dataset as speech
            train_dataset = speech.SiameseSpeechCommandsDataset
            test_dataset = speech.SpeechCommandsDataset
            bg_noise_dir = args.bg_noise_dir
            root = args.speech_commands_dir
        else:
            ValueError("Not supported dataset")

        dataset = [train_dataset, test_dataset]

    else:
        transform=[transforms.ToTensor(), transforms.ToTensor()],
        if args.dataset == "cifar10":
            dataset = datasets.CIFAR10
        elif args.dataset == "cifar100":  
            dataset = datasets.CIFAR100
        elif args.dataset == "svhn" or args.dataset == "svhn-core":  
            dataset = datasets.SVHN
            length = 20
        elif args.dataset == "imagenet":
            dataset = datasets.ImageNet
            root = args.imagenet_dir
            # fr CutMix https://arxiv.org/pdf/1905.04899.pdf
            length = 112
        elif args.dataset == "speech_commands":
            import dataset.speech_commands_dataset as speech
            dataset = speech.SpeechCommandsDataset
            bg_noise_dir = args.bg_noise_dir
            root = args.speech_commands_dir
        else:
            ValueError("Not supported dataset")


    if args.jitter:
        transform = color_jitter_transform()
    elif args.crop:
        transform = random_resized_crop_transform()
    else:
        transform = data_augment(dataset=args.dataset,
                                 length=length,
                                 cutout=args.cutout,
                                 auto_augment=args.auto_augment, 
                                 rand_augment=args.rand_augment, 
                                 rand_augment_mag=args.rand_augment_mag, 
                                 no_basic_augment=args.no_basic_augment,
                                 bg_noise_dir=bg_noise_dir,
                                 train_imagenet_size=args.train_imagenet_size,
                                 test_imagenet_size=args.test_imagenet_size)


    loader = DoubleLoader if args.agmax else SingleLoader

    dataloader = loader(root=root,
                        batch_size=args.batch_size, 
                        dataset=dataset,
                        transform=transform,
                        device=get_device(),
                        dataset_name=args.dataset,
                        shuffle_test=args.fgsm,
                        corruption=args.corruption,
                        num_workers=args.num_workers)

    backbone = backbones.get_backbone(dataset=args.dataset,
                                      n_classes=args.n_classes,
                                      pool_size=args.pool_size,
                                      feature_extractor=args.feature_extractor,
                                      backbone_config=args.backbone_config)


    classifier = net(args, 
                     backbone=backbone, 
                     dataloader=dataloader,
                     device=get_device())

    start_epoch = 1
    end_epoch = args.epochs + 1
    if args.resume:
        folder = args.checkpoints_dir
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, args.resume)
        print("Resuming from checkpoint '%s'" % path)
        checkpoint = torch.load(path)
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.model.to(get_device())
        classifier.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        classifier.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        last_epoch = checkpoint['epoch']
        classifier.assign_lr_scheduler(last_epoch=last_epoch-1)
        start_epoch = last_epoch + 1
        args.best_top1 = checkpoint['best_top1']
        args.best_top5 = checkpoint['best_top5']
        args.best_model = checkpoint['best_model']

        classifier.best_top1 = args.best_top1
        classifier.best_top5 = args.best_top5
        classifier.best_model = args.best_model
        classifier.best_epoch = checkpoint['best_epoch']

        if args.eval:
            val_name = None
            if args.corruption is not None:
                val_name = args.corruption
                print("Corruption mode:", args.corruption)
            return classifier.eval(start_epoch - 1, val_name=val_name)
        elif args.fgsm:
            epsilons = (0.1, 0.3, 0.5,)
            for eps in epsilons:
                ret = classifier.eval_robustness(eps, start_epoch - 1)

            return ret

        #print(classifier.model.module.feature_extractor)
        #print(classifier.model.module.backbone.feature_extractor)
        if args.save_extractor:
            if args.agmax:
                checkpoint = classifier.model.module.backbone.feature_extractor.state_dict()
            else:
                checkpoint = classifier.model.module.feature_extractor.state_dict()
            folder = args.checkpoints_dir
            os.makedirs(folder, exist_ok=True)
            filename = classifier.get_model_name() + "feature-extractor.pth"
            path = os.path.join(folder, filename)
            torch.save(checkpoint, path)
            print("Saving feature extractor: ", path)
            return None, None, None


    if args.train:
        best_top1 = args.best_top1
        best_top5 = args.best_top5
        best_model = args.best_model
        for epoch in range(start_epoch, end_epoch):
            start_time = datetime.datetime.now()
            loss = classifier.train(run, best_top1, best_top5, best_model, all_top1, epoch, label_smoothing=args.smoothing)
            top1, top5, model = classifier.eval(epoch)
            if args.dataset == "speech_commands":
                _, _, _ = classifier.eval(epoch, is_val=True)
            if args.plateau:
                classifier.scheduler.step(metrics=loss)
            else:
                classifier.scheduler.step(epoch)
            if top1 > best_top1:
                best_top1 = top1
                best_top5 = top5
                best_model = model
            elapsed_time = datetime.datetime.now() - start_time
            print("Elapsed time: %s" % elapsed_time)
        classifier._log(top1=top1, top5=top5, verbose=True)
        return top1, top5, model
    else:
        return None, None, None


if __name__ == '__main__':
    pass
