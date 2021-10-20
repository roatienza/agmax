python3 main.py --config=WideResNet28-10-auto_augment-cutmix-agmax --train --n-runs=1 --dataset=cifar100 --cutmix-prob=0.5
python3 main.py --config=WideResNet28-10-auto_augment-cutmix --train --n-runs=1 --dataset=cifar100 --cutmix-prob=0.5
python3 main.py --config=WideResNet28-10-cutmix-agmax --train --n-runs=1 --dataset=cifar100 --cutmix-prob=0.5
python3 main.py --config=WideResNet28-10-cutmix --train --n-runs=1 --dataset=cifar100 --cutmix-prob=0.5

python3 main.py --config=WideResNet28-10-auto_augment-mixup-agmax --train --n-runs=1 --dataset=cifar100 --alpha=1
python3 main.py --config=WideResNet28-10-auto_augment-mixup --train --n-runs=1 --dataset=cifar100 --alpha=1
python3 main.py --config=WideResNet28-10-mixup-agmax --train --n-runs=1 --dataset=cifar100 --alpha=1
python3 main.py --config=WideResNet28-10-mixup --train --n-runs=1 --dataset=cifar100 --alpha=1

python3 main.py --config=WideResNet28-10-auto_augment-cutout-agmax --train --n-runs=1 --dataset=cifar100
python3 main.py --config=WideResNet28-10-auto_augment-cutout --train --n-runs=1 --dataset=cifar100
python3 main.py --config=WideResNet28-10-cutout-agmax --train --n-runs=1 --dataset=cifar100
python3 main.py --config=WideResNet28-10-cutout --train --n-runs=1 --dataset=cifar100

python3 main.py --config=WideResNet28-10-standard --train --n-runs=1 --dataset=cifar100 --dropout=0.3
python3 main.py --config=WideResNet28-10-standard-agmax --train --n-runs=1 --dataset=cifar100 --dropout=0.3
python3 main.py --config=WideResNet28-10-standard --train --n-runs=1 --dataset=cifar100
python3 main.py --config=WideResNet28-10-standard-agmax --train --n-runs=1 --dataset=cifar100

python3 main.py --config=WideResNet28-10-auto_augment-agmax --train --n-runs=1 --dataset=cifar100
python3 main.py --config=WideResNet28-10-auto_augment --train --n-runs=1 --dataset=cifar100

#python3 main.py --config=WideResNet28-10-no_basic_augment-agmax --train --n-runs=1 --dataset=cifar100
#python3 main.py --config=WideResNet28-10-no_basic_augment --train --n-runs=1 --dataset=cifar100

