# ResNet101
python3 main.py --config=ResNet101-auto_augment-cutmix-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet101-auto_augment-cutmix --train --multisteplr --dataset=imagenet --epochs=270 --save 
python3 main.py --config=ResNet101-standard --train --multisteplr --dataset=imagenet --epochs=270 --save 


# RegNet and EfficientNet
CUDA_VISIBLE_DEVICES=0 python3 main.py --config=RegNetX002-auto_augment-cutmix-agmax  --train --cosinelr  --dataset=imagenet --save --n-units=2048 --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=1 python3 main.py --config=RegNetX002-auto_augment-cutmix  --train --cosinelr  --dataset=imagenet --save --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=2 python3 main.py --config=RegNetX002-standard-agmax --train --cosinelr  --dataset=imagenet --save --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=2 python3 main.py --config=RegNetX002-standard --train --cosinelr  --dataset=imagenet --save --test-imagenet-size=256

CUDA_VISIBLE_DEVICES=3 python3 main.py --config=EfficientNetB0-auto_augment-cutmix-agmax  --cosinelr --train --dataset=imagenet --save --n-units=2048 --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=4 python3 main.py --config=EfficientNetB0-auto_augment-cutmix  --cosinelr --train --dataset=imagenet --save  --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=5 python3 main.py --config=EfficientNetB0-standard-agmax  --cosinelr --train --dataset=imagenet --save  --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=5 python3 main.py --config=EfficientNetB0-standard  --cosinelr --train --dataset=imagenet --save  --test-imagenet-size=256

CUDA_VISIBLE_DEVICES=6 python3 main.py --config=RegNetY004-standard-agmax --train --cosinelr  --dataset=imagenet --save --n-units=2048 --test-imagenet-size=256
CUDA_VISIBLE_DEVICES=7 python3 main.py --config=EfficientNetB1-standard-agmax  --cosinelr --train --dataset=imagenet --save --n-units=2048 --test-imagenet-size=256
# 90 epochs
python3 main.py --config=ResNet50-standard-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save

# 270 epochs
python3 main.py --config=ResNet50-standard-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-cutmix-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-auto_augment-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-cutout-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-mixup-agmax --train --multisteplr --dataset=imagenet --epochs=270 --save

# 90 epochs
python3 main.py --config=ResNet50-standard --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-standard-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-mixup --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-mixup-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-cutmix --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-cutmix-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-cutout --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-cutout-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save

# 90 epochs auto augment
python3 main.py --config=ResNet50-auto_augment-cutout --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-cutout-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-cutmix --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-cutmix-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-mixup --train --multisteplr --dataset=imagenet --epochs=90 --save
python3 main.py --config=ResNet50-auto_augment-mixup-agmax --train --multisteplr --dataset=imagenet --epochs=90 --save

# 270 epochs standards
python3 main.py --config=ResNet50-auto_augment --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-standard --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-cutout --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-cutmix --train --multisteplr --dataset=imagenet --epochs=270 --save
python3 main.py --config=ResNet50-mixup --train --multisteplr --dataset=imagenet --epochs=270 --save


# feature extractor
python3 main.py --config=ResNet50-auto_augment-cutout   --dataset=imagenet  --checkpoints-dir weights/best/RestNet50-270 --resume imagenet-standard-ResNet50-cutout-auto_augment-77.86.pth  --save-extractor
