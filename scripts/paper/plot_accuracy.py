'''
Plot accuracy of AgMax results
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=imagenet --network=ResNet50-270epochs --vspace=0.6
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=imagenet --network=ResNet50-90epochs --vspace=0.6
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar10
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar10 --vspace=0.6
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar10 --network=WideResNet28-10  --vspace=0.6
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar10 --network=WideResNet28-10  --vspace=0.5
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar100 --network=WideResNet28-10  --vspace=0.5
(base) rowels-mbp:paper rowel$ python3 plot_accuracy.py --dataset=cifar100 --network=WideResNet40-2  --vspace=0.5
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import argparse

cifar10_WR40_2 = { 
            "Standard": { "Standard": "95.1", "+AgMax" : 95.6, "+AA": 95.9 , "+AA\n+AgMax": 96.4 },
            "CutOut": { "CutOut": "96.2", "+AgMax" : 96.6, "+AA": "96.4" , "+AA\n+AgMax": 97.1 },
            "MixUp": { "MixUp": 95.8, "+AgMax" : 96.3, "+AA": 96.0 , "+AA\n+AgMax": 96.6 },
            "CutMix": { "CutMix": 96.2, "+AgMax" : 96.7, "+AA": 96.4 , "+AA\n+AgMax": 96.8 },
            }

cifar10_WR28_10 = { 
            "Standard": { "Standard": "96.2", "+AgMax" : 96.4, "+AA": 96.9 , "+AA\n+AgMax": 97.4 },
            "CutOut": { "CutOut": "97.1", "+AgMax" : 97.3, "+AA": "97.5" , "+AA\n+AgMax": 97.8 },
            "MixUp": { "MixUp": "97.1", "+AgMax" : 97.5, "+AA": 97.4 , "+AA\n+AgMax": 97.9 },
            "CutMix": { "CutMix": 97.3, "+AgMax" : 97.7, "+AA": 97.4 , "+AA\n+AgMax": 97.8 },
            }

cifar100_WR40_2 = { 
            "Standard": { "Standard": "76.9", "+AgMax" : 77.1, "+AA": 78.4 , "+AA\n+AgMax": 79.0 },
            "CutOut": { "CutOut": "78.4", "+AgMax" : 79.2, "+AA": "80.0" , "+AA\n+AgMax": 81.0 },
            "MixUp": { "MixUp": 78.3, "+AgMax" : 79.5, "+AA": 79.1 , "+AA\n+AgMax": 80.7 },
            "CutMix": { "CutMix": 79.4, "+AgMax" : 80.0, "+AA": 80.1 , "+AA\n+AgMax": 81.3 },
            }

cifar100_WR28_10 = { 
            "Standard": { "Standard": "81.3", "+AgMax" : 81.7, "+AA": 82.8 , "+AA\n+AgMax": 83.0 },
            "CutOut": { "CutOut": "82.5", "+AgMax" : 82.6, "+AA": "83.8" , "+AA\n+AgMax": 84.0 },
            "MixUp": { "MixUp": "82.8", "+AgMax" : 82.9, "+AA": 84.1 , "+AA\n+AgMax": 84.8 },
            "CutMix": { "CutMix": 83.8, "+AgMax" : 84.0, "+AA": 85.0 , "+AA\n+AgMax": 85.3 },
            }

svhn_core_WR28_2 = { 
            "Standard": { "Standard": "96.8", "+AgMax" : 96.9, "+AA": 97.6 , "+AA\n+AgMax": 97.6 },
            "CutOut": { "CutOut": 97.4, "+AgMax" : 97.4, "+AA": "97.9" , "+AA\n+AgMax": 98.0 },
            "MixUp": { "MixUp": 97.3, "+AgMax" : 97.3, "+AA": 97.7 , "+AA\n+AgMax": 97.8 },
            "CutMix": { "CutMix": 97.4, "+AgMax" : 97.5, "+AA": 97.9 , "+AA\n+AgMax": 97.9 },
            }

svhn_WR28_2 = { 
            "Standard": { "Standard": "98.3", "+AgMax" : 98.4, "+AA": 98.5 , "+AA\n+AgMax": 98.5 },
            "CutOut": { "CutOut": 98.7, "+AgMax" : 98.8, "+AA": "98.7" , "+AA\n+AgMax": 98.8 },
            "MixUp": { "MixUp": 98.3, "+AgMax" : 98.4, "+AA": 98.4 , "+AA\n+AgMax": 98.5 },
            "CutMix": { "CutMix": 98.6, "+AgMax" : 98.6, "+AA": 98.7 , "+AA\n+AgMax": 98.7 },
            }

speech_LeNet5 = { 
            "Standard": { "Test": "89.9", "+AgMax" : 90.2, "Val": "90.0" , "+AgMax(Val)": 90.0 },
            "CutOut": { "Test": 89.0, "+AgMax" : 90.4, "Val": 89.2 , "+AgMax(Val)": 90.0 },
            "MixUp": { "Test": "89.4", "+AgMax" : 89.4, "Val": "89.4" , "+AgMax(Val)": 89.6 },
            "CutMix": { "Test": 87.1, "+AgMax" : 88.8, "Val": 87.5 , "+AgMax(Val)": 89.3 },
            }

speech_VGG11 = { 
            "Standard": { "Test": "96.3", "+AgMax" : 96.4, "Val": "96.0" , "+AgMax(Val)": 96.1 },
            "CutOut": { "Test": 96.5, "+AgMax" : 96.5, "Val": 96.1 , "+AgMax(Val)": 96.1 },
            "MixUp": { "Test": "96.5", "+AgMax" : 96.8, "Val": "96.2" , "+AgMax(Val)": 96.3 },
            "CutMix": { "Test": 96.4, "+AgMax" : 96.7, "Val": 96.2 , "+AgMax(Val)": 96.4 },
            }

imagenet_ResNet50_90_top1 = { 
            "Standard": { "Standard": "76.4", "+AgMax" : 76.9, "+AA": 76.2 , "+AA\n+AgMax": 77.1 },
            "CutOut": { "CutOut": 76.2, "+AgMax" : 77.1, "+AA": 75.7 , "+AA\n+AgMax": 76.6 },
            "MixUp": { "MixUp": "76.5", "+AgMax" : 77.6, "+AA": 75.9 , "+AA\n+AgMax": 77.1 },
            "CutMix": { "CutMix": 76.3, "+AgMax" : 77.4, "+AA": 75.5 , "+AA\n+AgMax": 77.0 },
            }

imagenet_ResNet50_90_top5 = { 
            "Standard": { "Standard": "93.2", "+AgMax" : 93.5, "+AA": 93.1 , "+AA\n+AgMax": 93.4 },
            "CutOut": { "CutOut": 93.1, "+AgMax" : 93.6, "+AA": 92.8 , "+AA\n+AgMax": 93.3 },
            "MixUp": { "MixUp": "93.3", "+AgMax" : 93.9, "+AA": 92.9 , "+AA\n+AgMax": 93.6 },
            "CutMix": { "CutMix": 93.2, "+AgMax" : 93.9, "+AA": 92.7 , "+AA\n+AgMax": 93.5 },
            }

imagenet_ResNet50_270_top1 = { 
            "Standard": { "Standard": "76.8", "+AgMax" : 77.2, "+AA": "77.6" , "+AA\n+AgMax": 77.7 },
            "CutOut": { "CutOut": "77.5", "+AgMax" : 77.6, "+AA": 77.9 , "+AA\n+AgMax": 78.2 },
            "MixUp": { "MixUp": "78.2", "+AgMax" : 78.4, "+AA": 78.3 , "+AA\n+AgMax": 78.6 },
            "CutMix": { "CutMix": "78.7", "+AgMax" : 79.0, "+AA": 78.5 , "+AA\n+AgMax": 79.1 },
            }

imagenet_ResNet50_270_top5 = { 
            "Standard": { "Standard": "93.3", "+AgMax" : 93.6, "+AA": "93.6" , "+AA\n+AgMax": 93.8 },
            "CutOut": { "CutOut": "93.6", "+AgMax" : 93.8, "+AA": 93.7 , "+AA\n+AgMax": 94.0 },
            "MixUp": { "MixUp": "93.9", "+AgMax" : 94.1, "+AA": 94.0 , "+AA\n+AgMax": 94.2 },
            "CutMix": { "CutMix": "94.2", "+AgMax" : 94.2, "+AA": 94.1 , "+AA\n+AgMax": 94.4 },
            }


agmax_vs_aa = {
        "CIFAR10\nWRN40-2": { "CutOut": 1.02, "MixUp": 0.70, "CutMix": 1.04, "AgMax": 0.48, "AA": 0.77, },
            "CIFAR10\nWRN28-10": { "CutOut": 0.87, "MixUp": 0.84, "CutMix": 1.07, "AgMax": 0.17, "AA": 0.69, },
            "CIFAR100\nWRN40-2": { "CutOut": 1.5, "MixUp": 1.42, "CutMix": 2.49, "AgMax": 0.49, "AA": 1.45, },
            "CIFAR100\nWRN28-10": { "CutOut": 1.17, "MixUp": 1.5, "CutMix": 2.51, "AgMax": 0.44, "AA": 1.50,  },
            "ImageNet\nRN50-90": {  "CutOut": -0.13, "MixUp": 0.06, "CutMix": -0.01, "AgMax": 0.43, "AA": -0.16, },
            "ImageNet\nRN50-270": {  "CutOut": 0.68, "MixUp": 1.41, "CutMix": 1.87, "AgMax": 0.39, "AA": 0.81, },
            "SVHN-core\nWRN28-2": {  "CutOut": 0.61, "MixUp": 0.49, "CutMix": 0.64, "AgMax": 0.08, "AA": 0.84,  },
            "SVHN\nWRN28-2": {  "CutOut": 0.41, "MixUp": 0.03, "CutMix": 0.27, "AgMax": 0.06, "AA": 0.16,  },
            "Speech\nLeNet5": { "CutOut": -0.84, "MixUp": -0.46, "CutMix": -2.81, "AgMax": 0.37, },
            "Speech\nVGG11": { "CutOut": 0.18, "MixUp": 0.22, "CutMix": 0.11, "AgMax": 0.13, },
        }

comparison = {
        "Data Augmentation": [ "CIFAR10\nWRN40-2", "CIFAR10\nWRN28-10", "CIFAR100\nWRN40-2", "CIFAR100\nWRN28-10", "ImageNet\nRN50-90", "ImageNet\nRN50-270", "SVHN-core\nWRN28-2", "SVHN\nWRN28-2", "Speech\nLeNet5", "Speech\nVGG11"   ],
        "CutOut" : [ 1.02, 0.87, 1.5, 1.17, -0.13,  0.68, 0.61, 0.41, -0.83,  0.15 ],
        "MixUp" : [0.70, 0.84, 1.42, 1.5, 0.14, 1.41, 0.49, 0.03, -0.54, 0.18, ],
        "CutMix" : [ 1.04, 1.07, 2.49, 2.51, -0.10, 1.87,  0.64, 0.27, -2.68, 0.14,  ],
        "AgMax" : [ 0.48,  0.17, 0.49, 0.44, 0.43, 0.39, 0.08, 0.06,  0.16,  0.09,  ],
        "AA" : [ 0.77, 0.69, 1.45, 1.50, -0.16, 0.81, 0.84, 0.16,   ],
        }

standard_batch = {
    "Batch Size" : [16,32,64,128,256,512],
    "Standard" :  [93.4,94.2,95.0,95.1,94.9,94.5],
    "Standard+AgMax" : [94.4,95.0,95.5,95.6,95.3,94.7],
    "Standard+AA" : [93.2,94.4,95.4,95.9,96.0,95.6],
    "Standard+AA+AgMax" : [94.5,95.7,96.1,96.4,96.2,95.8],
    }

cutout_batch = {
    "Batch Size" : [16,32,64,128,256,512],
    "CutOut": [92.7,95.1,95.7,96.2,96.1,96.0],
    "CutOut+AgMax": [94.9,95.9,96.4,96.6,96.5,95.9],
    "CutOut+AA" : [92.7,94.6,95.7,96.4,96.5,96.4],
    "CutOut+AA+AgMax" : [94.7,96.2,96.7,97.1,97.0,96.6],
    }

mixup_batch = {
    "Batch Size" : [16,32,64,128,256,512],
    "MixUp" : [92.8,94.7,95.4,95.8,95.9,95.8],
    "MixUp+AgMax" : [94.6,95.7,96.2,96.3,96.2,95.6],
    "MixUp+AA" : [92.5,94.0,95.2,96.0,96.2,96.3],
    "MixUp+AA+AgMax" : [94.5,95.7,96.3,96.6,96.6,96.4],
    }

cutmix_batch = {
    "Batch Size" : [16,32,64,128,256,512],
    "CutMix" : [93.4,94.9,95.8,96.2,96.2,95.7],
    "CutMix+AgMax" : [94.9,96.0,96.4,96.7,96.5,95.8],
    "CutMix+AA" : [93.0,94.4,96.0,96.4,96.4,96.3],
    "CutMix+AA+AgMax" : [94.8,96.0,96.6,96.8,96.9,96.4],
    }

batch_size = {
    "Batch Size" : [16,32,64,128,256,512],
    "Standard" : [94.4,95.0,95.5,95.6,95.3,94.7],
    "AA" : [94.5,95.7,96.1,96.4,96.2,95.8],
    "CutOut": [94.9,95.9,96.4,96.6,96.5,95.9],
    "CutOut+AA" : [94.7,96.2,96.7,97.1,97.0,96.6],
    "CutMix" : [94.9,96.0,96.4,96.7,96.5,95.8],
    "CutMix+AA" : [94.8,96.0,96.6,96.8,96.9,96.4],
    "MixUp" : [94.6,95.7,96.2,96.3,96.2,95.6],
    "MixUp+AA" :[94.5,95.7,96.3,96.6,96.6,96.4],
    }


standard_lr = {
    "Learning Rate" : [0.01,0.05,0.1,0.2,0.5],
    "Standard" :  [94.0,95.1,95.1,95.0,93.1 ],
    "Standard+AgMax" : [91.5,95.3,95.6,95.3,94.2 ],
    "Standard+AA" : [95.2,95.9,95.9,95.2,93.4 ],
    "Standard+AA+AgMax" : [93.7,96.2,96.4,96.3,94.6 ],
    }

cutout_lr = {
    "Learning Rate" : [0.01,0.05,0.1,0.2,0.5],
    "CutOut": [95.5,96.2,96.2,95.6,93.2 ],
    "CutOut+AgMax": [93.9,96.3,96.6,96.4,94.9 ],
    "CutOut+AA" : [95.9,96.5,96.4,95.9,93.3 ],
    "CutOut+AA+AgMax" : [94.8,96.9,97.1,96.8,95.3 ],
    }

mixup_lr = {
    "Learning Rate" : [0.01,0.05,0.1,0.2,0.5],
    "MixUp" : [95.2,95.8,95.8,95.0,93.1 ],
    "MixUp+AgMax" : [93.1,96.3,96.3,96.0,94.8 ],
    "MixUp+AA" : [95.8,96.0,96.0,95.4,92.9 ],
    "MixUp+AA+AgMax" : [93.4,96.7,96.6,96.2,94.9 ],
    }

cutmix_lr = {
    "Learning Rate" : [0.01,0.05,0.1,0.2,0.5],
    "CutMix" : [95.2,96.1,96.2,95.5,93.1 ],
    "CutMix+AgMax" : [92.7,96.5,96.7,96.4,94.5 ],
    "CutMix+AA" : [95.8,96.4,96.4,95.7,93.1 ],
    "CutMix+AA+AgMax" : [93.7,96.9,96.8,96.5,94.6 ],
    }

def plot_line(dataset, filename="fig.png", ylabel="Top-1% Accuracy"):
    data = {}
    keys = []
    fig, ax = plt.subplots()
    plt.xlabel('xlabel', fontsize=14)
    plt.ylabel('ylabel', fontsize=14) 
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=11)
    plt.ylabel(ylabel)
    markers = ['^', 's', 'o', 'D', 'X', '<', 'P', 'X']
    colors = sns.color_palette()[0:]
    i = 0
    j = 0
    linestyle = True
    is_aa = False
    fontsize = 11
    spacing = 0.8
    for key, val in dataset.items():
        #val = list(val)
        if "Batch" in key:
            xlabel = key
            xval = val
            ax.set_xscale('log', basex=2)
            ax.set_xticks(xval)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.xlabel(key)
            continue
        elif "Rate" in key:
            xlabel = key
            xval = val
            ax.set_xscale('log', basex=2)
            ax.set_xticks(xval)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(x, '')))
            plt.xlabel(key)
            continue
        elif "Data Augmentation" in key:
            xlabel = key
            xval = val
            ax.set_xticklabels(xval, rotation=90)
            ax.grid(True)
            plt.xticks([m for m in range(len(xval))], xval)
            linestyle = False
            fig.subplots_adjust(bottom=0.2)
            plt.xlabel('')
            fontsize = 12
            spacing = 1.5
            continue
        elif "AA" in key and not linestyle:
            is_aa = True
            plt.plot(xval[0:-2], val, linewidth=2, color=colors[j], marker=markers[j], markersize=10, label=key)
            continue
        else:
            data[key] = val 

        if i%2 > 0:
            plt.plot(xval, val, linewidth=2, color=colors[j], marker=markers[j], markersize=10, label=key)
            j += 1
        elif linestyle:
            plt.plot(xval, val, linewidth=2, color=colors[j], marker=markers[j], markersize=10, label=key, linestyle='dashed', fillstyle='none')
        else:
            plt.plot(xval, val, linewidth=2, color=colors[j], marker=markers[j], markersize=10, label=key)
            j += 1
        i += 1

    ax.legend(loc='upper center', fontsize=fontsize, ncol=i+1, bbox_to_anchor=(0.5, 1.15), frameon=False, columnspacing=spacing, handletextpad=0.2)

    plt.savefig(filename)
    plt.show()

def plot_bar(dataset, title, ylabel="Top-1% Accuracy", xlabel="Data Augmentation", ncolors=4, vspace=0.2, is_scatter=False):
    plt.rc('font', size=12) 
    if not is_scatter:
        plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=16)
    if ncolors == 4:
        plt.rc('xtick', labelsize=14)
    else:
        plt.rc('xtick', labelsize=11)

    fig, ax = plt.subplots()
    if not is_scatter:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not is_scatter:
        ax.set_title(title)
    colors = sns.color_palette()[0:ncolors]
    markers = ['^', 's', 'o', 'D', '*']

    xticks = len(dataset)
    x = np.arange(xticks)
    ax.set_xticks(x)
    if ncolors == 4:
        width = 0.2
    else:
        fig.subplots_adjust(bottom=0.2)
        width = 0.1
        #ax.axhline(y=0, color='gray')

    i = 0
    j = 0
    low = 100
    high = 0
    labels = []
    has_label = False
    for key, val in dataset.items():
        for k, v in val.items():
            if isinstance(v, str):
                v = float(v)
                color = "gray"
            else:
                color = colors[i]

            if v < low:
                low = v
            if v > high:
                high = v
        
            if ncolors == 4:
                x = j-((1 - i)*width+width/2)
            else:
                x = j-((1 - i)*width+width/2)
                #x = j-(i*width - 0.2)

            #if "Speech" in key:
            #    tcolors = sns.color_palette()[0:5]
            #    if is_scatter:
            #        ax.scatter(j, v, marker=markers[i], s=100, color=tcolors[i+2])
            #    else:
            #        ax.bar(x, v, width, color=tcolors[i+2])
            #else:
            if is_scatter:
                if not has_label:
                    ax.scatter(j, v, marker=markers[i], s=100, label=k, color=color)
                else:
                    ax.scatter(j, v, marker=markers[i], s=100, color=color)
            else:
                ax.bar(x, v, width, color=color)

            height = v
            offset = 3
            if v < 0:
                height = 0
                

            if is_scatter is False: 
                ax.annotate('{}'.format(k),
                            xy=(x, height),
                            xytext=(0, offset),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            rotation=90)
            #ax.annotate(k, (key, v))
            i += 1
            i %= len(colors)

        labels.append(key)
        j += 1
        has_label = True
    
    if is_scatter:
        ax.legend(loc='upper center', ncol=ncolors, bbox_to_anchor=(0.5, 1.12), frameon=False, columnspacing=1, handletextpad=0.0)
        ax.grid(True)

    if ncolors == 4:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels(labels, rotation=90)
    import math
    if "100" in title:
        plt.ylim([low-0.2, high+1.1])
    else:
        # change high+k manually
        plt.ylim([low-0.2, high+vspace])
    title = title.replace(" ", "_")
    title = title.replace("%", "")
    plt.savefig(title + ".png")
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgMax Results')
    parser.add_argument('--dataset',
                        default=None,
                        help='Dataset to plot results')
    parser.add_argument('--network',
                        default="WideResNet40-2",
                        help='Network name')
    parser.add_argument('--top',
                        default="top1",
                        help='Top-1 or -5%')
    parser.add_argument('--vspace',
                        default=0.2,
                        type=float,
                        help='Vertical space in bar graph label')
    args = parser.parse_args()
    ylabel = "Top-1% Accuracy"
    xlabel = "Data Augmentation"
    ncolors = 4
    is_scatter = False

    if args.dataset == "cifar10":
        title = "CIFAR10 "
        if args.network == "WideResNet40-2":
            dataset = cifar10_WR40_2
            title += args.network
            title += " 200 epochs"
        else:
            dataset = cifar10_WR28_10
            title += args.network
            title += " 200 epochs"
    elif args.dataset == "cifar100":
        title = "CIFAR100 "
        if args.network == "WideResNet40-2":
            dataset = cifar100_WR40_2
            title += args.network
            title += " 200 epochs"
        else:
            dataset = cifar100_WR28_10
            title += args.network
            title += " 200 epochs"
    elif args.dataset == "svhn-core":
        title = "SVHN-core WideResNet28-2"
        dataset = svhn_core_WR28_2 
        title += " 200 epochs"
    elif args.dataset == "svhn":
        title = "SVHN WideResNet28-2"
        dataset = svhn_WR28_2 
        title += " 160 epochs"
    elif args.dataset == "speech-lenet5":
        title = "Speech Commands LeNet5"
        dataset = speech_LeNet5 
        title += " 30 epochs"
    elif args.dataset == "speech-vgg11":
        title = "Speech Commands VGG11"
        dataset = speech_VGG11
        title += " 30 epochs"
        print(args.dataset)
    elif args.dataset == "imagenet":
        title = "ImageNet"
        if args.network == "ResNet50-90epochs":
            if args.top == "top1":
                dataset = imagenet_ResNet50_90_top1
                title += " ResNet50 90 epochs Top-1%"
            else:
                dataset = imagenet_ResNet50_90_top5
                title += " ResNet50 90 epochs Top-5%"
                ylabel="Top-5% Accuracy"
        else:
            # python3 plot_accuracy.py --dataset=imagenet --network=ResNet50-270epochs
            # python3 plot_accuracy.py --dataset=imagenet --network=ResNet50-270epochs --top=top5
            if args.top == "top1":
                dataset = imagenet_ResNet50_270_top1
                title += " ResNet50 270 epochs Top-1%"
            else:
                dataset = imagenet_ResNet50_270_top5
                title += " ResNet50 270 epochs Top-5%"
                ylabel="Top-5% Accuracy"
    elif args.dataset == "batch":
        plot_line(dataset=batch_size, filename="batch.png")
        exit(0)
    elif args.dataset == "standard_batch":
        plot_line(dataset=standard_batch, filename="standard_batch.png")
        exit(0)
    elif args.dataset == "cutout_batch":
        plot_line(dataset=cutout_batch, filename="cutout_batch.png")
        exit(0)
    elif args.dataset == "mixup_batch":
        plot_line(dataset=mixup_batch, filename="mixup_batch.png")
        exit(0)
    elif args.dataset == "cutmix_batch":
        plot_line(dataset=cutmix_batch, filename="cutmix_batch.png")
        exit(0)
    elif args.dataset == "comparison":
        plot_line(dataset=comparison, filename="comparison.png", ylabel="Top-1% Accuracy Increase")
        exit(0)
    elif args.dataset == "standard_lr":
        plot_line(dataset=standard_lr, filename="standard_lr.png")
        exit(0)
    elif args.dataset == "cutout_lr":
        plot_line(dataset=cutout_lr, filename="cutout_lr.png")
        exit(0)
    elif args.dataset == "mixup_lr":
        plot_line(dataset=mixup_lr, filename="mixup_lr.png")
        exit(0)
    elif args.dataset == "cutmix_lr":
        plot_line(dataset=cutmix_lr, filename="cutmix_lr.png")
        exit(0)
    elif args.dataset is None:
        title = "Average % Accuracy Improvement"
        dataset = agmax_vs_aa
        ylabel = "Top-1% Accuracy Increase"
        xlabel = "Dataset-Model"
        ncolors = 5
        is_scatter = True


    plot_bar(dataset=dataset, title=title, ylabel=ylabel, xlabel=xlabel, ncolors=ncolors, vspace=args.vspace, is_scatter=is_scatter)
    #plot_results(dataset=args.dataset, title=args.title)
    #plot_gaussian_1d(args.mean, args.std, args.model)
