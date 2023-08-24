#%%
import torch
from torch import nn
# from torch import nn, autograd
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
from models import *
import numpy as np

from utils.getData import *
import argparse
import os
import torchsummary
from math import sqrt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100)
parser.add_argument('--noniid', action='store_true') # default: false
parser.add_argument('--class_per_each_client', type=int, default=10)

# parser.add_argument('--frac', type=float, default=0.1)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--local_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--local_ep', type=int, default=5)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--rs', type=int, default=0)

parser.add_argument('--min_flex_num', type=int, default=2, help="0:0~ max(0,tc-args.min_flex_num)")
parser.add_argument('--max_flex_num', type=int, default=2, help="0:~4 min(tc+args.max_flex_num+1,5)")

parser.add_argument('--num_experiment', type=int, default=3, help="the number of experiments")
parser.add_argument('--model_name', type=str, default='resnet34') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--learnable_step', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)
parser.add_argument('--model_num', type=int, default=5)
parser.add_argument('--method', type=str, default='OD')
parser.add_argument('--plot', type=str, default='steps') # l1norm, steps

    
# args = parser.parse_args()
args = parser.parse_args(args=[])
args.device = 'cuda:' + args.device_id

X = [[], [], [], [], []]
Xk = []
BNs = [[], [], [], [], []]
Steps = [[], [], [], [], []]

args.ps, args.s2D = get_submodel_info(args)

local_models = []
ff = '20230815-203354[cifar10][resnet34]NeFLADD[iid]0'
# ff = '20230815-203413[cifar10][resnet34]NeFLAOD[iid]0'

# filename = './output/nefl/' + '20230814-091553[cifar10][resnet18]NeFLADD[iid]0' + '/models'
# filename = './output/nefl/' + '20230814-091622[cifar10][Presnet18]NeFLADD[iid]0' + '/models'
# filename = './output/nefl/' + '20230814-131602[cifar10][resnet18]NeFLW[iid]0' + '/models'
filename = './output/nefl/' + ff + '/models'

if args.model_name == 'resnet18':
    for i in range(len(args.ps)):
        # tmp_model = resnet18wd(args.s2D[-1][0], args.ps[i], True, num_classes=args.num_classes)
        tmp_model = resnet18wd(args.s2D[i][0], args.ps[i], True, num_classes=args.num_classes)
        tmp_model.load_state_dict(torch.load(os.path.join(filename, 'model' + str(i) + '.pt')))
        local_models.append(tmp_model)
elif args.model_name == 'resnet34':
    for i in range(len(args.ps)):
        tmp_model = resnet34wd(args.s2D[i][0], args.ps[i], True, num_classes=args.num_classes)
        tmp_model.load_state_dict(torch.load(os.path.join(filename, 'model' + str(i) + '.pt')))
        local_models.append(tmp_model)
        
    for k in tmp_model.state_dict():
        # print(k)
        if args.plot == 'steps':
            if 'step' in k:
                for i in range(len(args.ps)):
                    x = local_models[i].state_dict()[k] # 'conv1.weight' 'layer1.0.conv1.weight'
                    # print(sum(abs(x))/len(x))
                    Steps[i].append(x)
        elif args.plot == 'l1norm':
            if 'num_batches' not in k and 'bn' not in k and 'step' not in k:
                Xk.append(k)
                for i in range(len(args.ps)):
                    x = local_models[i].state_dict()[k].reshape(-1) # 'conv1.weight' 'layer1.0.conv1.weight'
                    # print(sum(abs(x))/len(x))
                    X[i].append(sum(abs(x))/len(x)) # Avg. L1 norm
        # elif 'num_batches' not in k and 'bn' in k:
        #     for i in range(len(args.ps)):
        #         x = local_models[i].state_dict()[k].reshape(-1) # 'conv1.weight' 'layer1.0.conv1.weight'
        #         # print(sum(abs(x))/len(x))
        #         BNs[i].append(sum(abs(x))/len(x))  # Avg. L1 norm

t = np.arange(0,len(X[0]),1)
t2 = np.arange(0,len(BNs[0]),1)
t3 = np.arange(1,len(Steps[0])+1,1)

plt.rcParams.update({'font.size': 15})
# plt.plot(t, X[0], 'r--', t, X[1], 'g-.', t, X[2], 'bs:', t, X[3], 'y^-', t, X[4], 'k')
# plt.plot(t2, BNs[0], 'r', t2, BNs[1], 'g', t2, BNs[2], 'b', t2, BNs[3], 'y', t2, BNs[4], 'k')
if args.plot == 'l1norm':
    plt.xlabel('Layers')
    plt.ylabel('Avg. L1 norm')
    plt.plot(t, X[0], 'r--', label = 'Submodel 1')
    plt.plot(t, X[1], 'g-.', label = 'Submodel 2')
    plt.plot(t, X[2], 'bs:', label = 'Submodel 3')
    plt.plot(t, X[3], 'y^-', label = 'Submodel 4')
    plt.plot(t, X[4], 'k', label = 'Submodel 5')
elif args.plot == 'steps':
    plt.xlabel('Blocks')
    plt.ylabel('Trained step size')
    plt.plot(t3, Steps[0], 'r', label = 'Submodel 1')
    plt.plot(t3, Steps[1], 'g', label = 'Submodel 2')
    plt.plot(t3, Steps[2], 'b', label = 'Submodel 3')
    plt.plot(t3, Steps[3], 'y', label = 'Submodel 4')
    plt.plot(t3, Steps[4], 'k', label = 'Submodel 5')
plt.legend(loc='upper left')
plt.show()
print('Models Loaded')
plt.savefig('myFigure.pdf')
# %%
