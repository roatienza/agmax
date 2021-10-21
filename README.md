# Improving Model Generalization by Agreement of Learned Representations from Data Augmentation (WACV 2022)


## Why it matters?

When data augmentation is applied on an input image, a model is forced to learn the correct label to improve model generalization (Figure 1). 

<img src="https://github.com/roatienza/agmax/blob/master/figures/fig1_agmax.png" width="50%" height="50%">


Since data augmentation incurs little overhead, why not generate 2 data augmented images (also known as 2 positive samples) from a given input. Then, force the model to agree on the correct label (Figure 2). It turns that maximizing this agreement further improves model model generalization. We call our method *AgMax*.


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


### Paper

Soon.

### Install requirements

```
pip3 install -r requirements.txt
```

### Train

For example, train ResNet50 with Agmax on 2 GPUs for 90 epochs, SGD with `lr=0.1` and multistep learning rate:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=ResNet50-standard-agmax --train --multisteplr \
--dataset=imagenet --epochs=90 --save
```

Compare the results without Agmax:

```
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config=ResNet50-standard --train --multisteplr \
--dataset=imagenet --epochs=90 --save
```

### Test


```
```

