#%%
import torch
# from torch import nn, autograd
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import torch
import copy
from models import *
from fed import *
import numpy as np

from torchvision import datasets, transforms
from torchvision.models import resnet18 as Presnet18
from torchvision.models import resnet34 as Presnet34
from torchvision.models import resnet101 as Presnet101
from torchvision.models import wide_resnet101_2 as Pwide_resnet101_2
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet101_Weights, Wide_ResNet101_2_Weights

from getData import *
import argparse
import os
import torchsummary
from math import sqrt
import wandb
from datetime import datetime

from utils.util import test_img, extract_submodel_weight_from_globalM, get_logger
from utils.avg_temp import MAAvg

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
parser.add_argument('--model_name', type=str, default='resnet18') # wide_resnet101_2
parser.add_argument('--device_id', type=str, default='0')
parser.add_argument('--learnable_step', type=bool, default=True)
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--wandb', type=bool, default=True)

parser.add_argument('--name', type=str, default='[MAFL][R18-k3]')

    
# args = parser.parse_args()
args = parser.parse_args(args=[])

args.device = 'cuda:' + args.device_id

X = [[], [], [], [], []]
Xk = []
BNs = [[], [], [], [], []]
Steps = [[], [], [], [], []]

args.ps = [sqrt(0.2), sqrt(0.4), sqrt(0.6), sqrt(0.8), 1]
args.s2D = [ # 18
        [ [[1, 1], [1, 1], [1, 1], [1, 1]] ],
        [ [[1, 1], [1, 1], [1, 1], [1, 1]] ],
        [ [[1, 1], [1, 1], [1, 1], [1, 1]] ],
        [ [[1, 1], [1, 1], [1, 1], [1, 1]] ],
        [ [[1, 1], [1, 1], [1, 1], [1, 1]] ]
    ]
args.num_models = len(args.ps)

local_models = []
if args.model_name == 'resnet18':
    for i in range(len(args.ps)):
        tmp_model = resnet18Mp(args.s2D[i][0], args.num_classes, args.ps[i])
        filename = '/data/output/mafl/'+ '20230323-101719' + str(args.name) + str(args.rs) + '/models'
        tmp_model.load_state_dict(torch.load(os.path.join(filename, 'model' + str(i) + '.pt')))
        local_models.append(tmp_model)
        
    for k in tmp_model.state_dict():
        print(k)
        if 'num_batches' not in k and 'bn' not in k and 'step' not in k:
            Xk.append(k)
            for i in range(len(args.ps)):
                x = local_models[i].state_dict()[k].reshape(-1) # 'conv1.weight' 'layer1.0.conv1.weight'
                # print(sum(abs(x))/len(x))
                X[i].append(sum(abs(x))/len(x))
        elif 'num_batches' not in k and 'bn' in k:
            for i in range(len(args.ps)):
                x = local_models[i].state_dict()[k].reshape(-1) # 'conv1.weight' 'layer1.0.conv1.weight'
                # print(sum(abs(x))/len(x))
                BNs[i].append(sum(abs(x))/len(x))
        elif 'step' in k:
            for i in range(len(args.ps)):
                x = local_models[i].state_dict()[k].reshape(-1) # 'conv1.weight' 'layer1.0.conv1.weight'
                # print(sum(abs(x))/len(x))
                Steps[i].append(sum(abs(x))/len(x))                

t = np.arange(0,len(X[0]),1)
t2 = np.arange(0,len(BNs[0]),1)
t3 = np.arange(0,len(Steps[0]),1)

plt.plot(t, X[0], 'r--', t, X[1], 'g-.', t, X[2], 'bs:', t, X[3], 'y^-', t, X[4], 'k')
plt.xlabel('layer')
plt.ylabel('L1 norm')
plt.legend(loc='upper left')
# plt.plot(t2, BNs[0], 'r', t2, BNs[1], 'g', t2, BNs[2], 'b', t2, BNs[3], 'y', t2, BNs[4], 'k')
# plt.plot(t3, Steps[0], 'r', t3, Steps[1], 'g', t3, Steps[2], 'b', t3, Steps[3], 'y', t3, Steps[4], 'k')
plt.show()
print('Models Loaded')
# %%
 