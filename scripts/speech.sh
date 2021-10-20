#LeNet
python3 main.py --config=LeNet-standard-agmax --train --multisteplr --adam   --dataset=speech_commands --epochs=30
python3 main.py --config=LeNet-mixup-agmax --train --multisteplr --adam   --dataset=speech_commands --epochs=30 --alpha=0.1
python3 main.py --config=LeNet-cutout-agmax --train --multisteplr --adam   --dataset=speech_commands --epochs=30
python3 main.py --config=LeNet-cutmix-agmax --train --multisteplr --adam   --dataset=speech_commands --epochs=30 --cutmix-prob=0.5

python3 main.py --config=LeNet-standard --train --multisteplr --adam   --dataset=speech_commands --epochs=30
python3 main.py --config=LeNet-mixup --train --multisteplr --adam   --dataset=speech_commands --epochs=30 --alpha=0.1
python3 main.py --config=LeNet-cutout --train --multisteplr --adam   --dataset=speech_commands --epochs=30
python3 main.py --config=LeNet-cutmix --train --multisteplr --adam   --dataset=speech_commands --epochs=30 --cutmix-prob=0.5

#VGG11
python3 main.py --config=VGG11-standard-agmax --train --multisteplr --dataset=speech_commands --adam --epochs=30
python3 main.py --config=VGG11-mixup-agmax --train --multisteplr --dataset=speech_commands  --adam --epochs=30
python3 main.py --config=VGG11-cutout-agmax --train --multisteplr --dataset=speech_commands  --adam --epochs=30
python3 main.py --config=VGG11-cutmix-agmax --train --multisteplr --dataset=speech_commands  --adam --epochs=30  --cutmix-prob=0.5

python3 main.py --config=VGG11-standard --train --multisteplr --dataset=speech_commands  --adam --epochs=30
python3 main.py --config=VGG11-mixup --train --multisteplr --dataset=speech_commands  --adam --epochs=30
python3 main.py --config=VGG11-cutout --train --multisteplr --dataset=speech_commands  --adam --epochs=30
python3 main.py --config=VGG11-cutmix --train --multisteplr --dataset=speech_commands  --adam --epochs=30  --cutmix-prob=0.5
