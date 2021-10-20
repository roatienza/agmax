python3 main.py --config=WideResNet40-2-auto_augment-cutmix-agmax --train --n-runs=1 --dataset=cifar10  --cutmix-prob=0.5
python3 main.py --config=WideResNet40-2-auto_augment-cutmix --train --n-runs=1 --dataset=cifar10 --cutmix-prob=0.5
python3 main.py --config=WideResNet40-2-cutmix-agmax --train --n-runs=1 --dataset=cifar10 --cutmix-prob=0.5
python3 main.py --config=WideResNet40-2-cutmix --train --n-runs=1 --dataset=cifar10 --cutmix-prob=0.5

python3 main.py --config=WideResNet40-2-auto_augment-mixup-agmax --train --n-runs=1 --dataset=cifar10 --alpha=1
python3 main.py --config=WideResNet40-2-auto_augment-mixup --train --n-runs=1 --dataset=cifar10 --alpha=1
python3 main.py --config=WideResNet40-2-mixup-agmax --train --n-runs=1 --dataset=cifar10 --alpha=1
python3 main.py --config=WideResNet40-2-mixup --train --n-runs=1 --dataset=cifar10 --alpha=1

python3 main.py --config=WideResNet40-2-auto_augment-cutout-agmax --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-auto_augment-cutout --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-cutout-agmax --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-cutout --train --n-runs=1 --dataset=cifar10

python3 main.py --config=WideResNet40-2-no_basic_augment-agmax --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-no_basic_augment --train --n-runs=1 --dataset=cifar10

python3 main.py --config=WideResNet40-2-standard --train --n-runs=1 --dataset=cifar10 --dropout=0.3
python3 main.py --config=WideResNet40-2-standard-agmax --train --n-runs=1 --dataset=cifar10 --dropout=0.3
python3 main.py --config=WideResNet40-2-standard --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-standard-agmax --train --n-runs=1 --dataset=cifar10

python3 main.py --config=WideResNet40-2-auto_augment-agmax --train --n-runs=1 --dataset=cifar10
python3 main.py --config=WideResNet40-2-auto_augment --train --n-runs=1 --dataset=cifar10

