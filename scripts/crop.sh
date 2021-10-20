python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-standard --resume=imagenet-standard-ResNet50-76.82-mlp-2048.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-cutmix --resume=imagenet-standard-ResNet50-cutmix-78.69.pth 
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-cutout --resume=imagenet-standard-ResNet50-cutout-77.50-mlp-2048.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-mixup --resume=imagenet-standard-ResNet50-mixup-78.23.pth

python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-cutmix-agmax --resume=imagenet-agmax-ResNet50-cutmix-78.97-mlp-2048.pth 
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-cutmix --resume=imagenet-standard-ResNet50-cutmix-auto_augment-78.5.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-cutmix-agmax --resume=imagenet-agmax-ResNet50-cutmix-auto_augment-79.12-mlp-2048.pth 

python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-cutout-agmax --resume=imagenet-agmax-ResNet50-cutout-77.61-mlp-2048.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-cutout --resume=imagenet-standard-ResNet50-cutout-auto_augment-77.86.pth 
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-cutout-agmax --resume=imagenet-agmax-ResNet50-cutout-auto_augment-78.17-mlp-2048.pth

python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-mixup-agmax --resume=imagenet-agmax-ResNet50-mixup-78.37-mlp-2048.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-mixup --resume=imagenet-standard-ResNet50-mixup-auto_augment-78.31.pth 
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-mixup-agmax --resume=imagenet-agmax-ResNet50-mixup-auto_augment-78.61-mlp-2048.pth

python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-standard-agmax --resume=imagenet-agmax-ResNet50-77.16-mlp-2048.pth
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment --resume=imagenet-standard-ResNet50-auto_augment-77.63-mlp-2048.pth 
python3 main.py --dataset=imagenet --checkpoints-dir=weights/best/270  --batch-size=32 --crop --eval --config=ResNet50-auto_augment-agmax --resume=imagenet-agmax-ResNet50-auto_augment-77.67-mlp-2048.pth

