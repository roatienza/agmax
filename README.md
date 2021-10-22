# Improving Model Generalization by Agreement of Learned Representations from Data Augmentation (WACV 2022)

### Paper

[ArXiv](https://arxiv.org/abs/2110.10536)

## Why it matters?

When data augmentation is applied on an input image, a model is forced to learn invariant features to improve model generalization (Figure 1). 

<img src="https://github.com/roatienza/agmax/blob/master/figures/fig1_agmax.png" width="50%" height="50%">


Since data augmentation incurs little overhead, why not generate 2 data augmented images (also known as 2 positive samples) from a given input. Then, force the model to agree on the common invariant features to support the correct label (Figure 2). It turns out that maximizing this agreement further improves model model generalization. We call our method *AgMax*.


<img src="https://github.com/roatienza/agmax/blob/master/figures/fig2_agmax.png" width="50%" height="50%">

Unlike label smoothing, AgMax consistently improves model accuracy. For example on ImageNet1k for 90 epochs, the ResNet50 performance is as follows:


| Data Augmentation | Baseline | Label Smoothing | AgMax (Ours) |
| :------------ | :-------------: | :-------------: | :-------------: |
| Standard | 76.4 | 76.8 | 76.9 | 
| CutOut | 76.2 | 76.5 | 77.1 |
| MixUp | 76.5 | 76.7| **77.6**  |
| CutMix | 76.3 | 76.4 | 77.4 |
| AutoAugment (AA) | 76.2 | 76.2 | 77.1 |
| CutOut+AA | 75.7 | 75.7 | 76.6 |
| MixUp+AA | 75.9 | 76.5 | 77.1 |
| CutMix+AA | 75.5 | 75.5 | 77.0 |

The figure below demonstrates consistent improvement across different data augmnentation methods:

<img src="https://github.com/roatienza/agmax/blob/master/figures/ImageNet_ResNet50_90_epochs_Top-1.png" width="50%" height="50%">



### Install requirements

```
pip3 install -r requirements.txt
```

### Train

For example, train ResNet50 with AgMax on 2 GPUs for 90 epochs, SGD with `lr=0.1` and multistep learning rate scheduler:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=ResNet50-standard-agmax --train \
--multisteplr --dataset=imagenet --epochs=90 --save
```

Compare the results without AgMax:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=ResNet50-standard --train \
--multisteplr --dataset=imagenet --epochs=90 --save
```

### Test

Using a pre-trained model:

ResNet101 trained with CutMix, AutoAugment and AgMax:
```
mkdir checkpoints
cd checkpoints
wget https://github.com/roatienza/agmax/releases/download/agmax-0.1.0/imagenet-agmax-mi-ResNet101-cutmix-auto_augment-81.19-mlp-4096.pth
cd ..
python3 main.py --config=ResNet101-auto_augment-cutmix-agmax --eval \
--dataset=imagenet \
--resume imagenet-agmax-mi-ResNet101-cutmix-auto_augment-81.19-mlp-4096.pth
```

ResNet50 trained with CutMix, AutoAugment and AgMax:

```
python3 main.py --config=ResNet50-auto_augment-cutmix-agmax --eval --n-units=2048 \
--dataset=imagenet --resume imagenet-agmax-ResNet50-cutmix-auto_augment-79.12-mlp-2048.pth
```

Other pre-trained models (Baselines):

- [ResNet101 Cutmix+AutoAugment](https://github.com/roatienza/agmax/releases/download/agmax-0.1.0/imagenet-standard-ResNet101-cutmix-auto_augment-80.69.pth)
- [ResNet50 Cutmix+AutoAugment](https://github.com/roatienza/agmax/releases/download/agmax-0.1.0/imagenet-standard-ResNet50-cutmix-auto_augment-78.5.pth)

## Citation
If you find this work useful, please cite:

```
@inproceedings{atienza2022agmax,
  title={Improving Model Generalization by Agreement of Learned Representations from Data Augmentation},
  author={Atienza, Rowel},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2022},
  pubstate={published},
  tppubtype={inproceedings}
}
```
